import hashlib
import re

import numpy as np

from config import EnvironmentConfig, TrainConfig
from utils.logger import setup_custom_logger

logger = setup_custom_logger('ENCODER')

DATE_LITERAL_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
NUMBER_LITERAL_RE = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)")


class UnifiedFeatureEncoder:
    def __init__(self, enable_predicate_encoding=None):
        self.op_list = ['Aggregate', 'Scan', 'Join', 'LogicalQueryStage', 'AQEShuffleRead', 'Exchange', 'BroadcastExchange']
        with open(EnvironmentConfig.table_file) as f:
            self.table_list = [s.strip() for s in f.readlines()]

        self.enable_predicate_encoding = (
            TrainConfig.enable_predicate_encoding
            if enable_predicate_encoding is None
            else enable_predicate_encoding
        )

        self.column_bucket_count = 8
        self.operator_stat_count = 5
        self.literal_stat_count = 6
        self.predicate_dim = (
            self.column_bucket_count + self.operator_stat_count + self.literal_stat_count
        )
        self.base_features = 1 + 8 + len(self.table_list) + 8 + 2
        self.in_features = self.base_features + (
            self.predicate_dim if self.enable_predicate_encoding else 0
        )

    def _stable_bucket(self, key: str, bucket_count: int) -> int:
        digest = hashlib.sha256(key.encode('utf-8')).digest()
        return int.from_bytes(digest[:8], byteorder='big', signed=False) % bucket_count

    def _classify_literal(self, value):
        if value is None:
            return None

        text = str(value).strip()
        if not text:
            return None
        if DATE_LITERAL_RE.fullmatch(text):
            return 'date'
        if NUMBER_LITERAL_RE.fullmatch(text):
            return 'num'
        return 'str'

    def _encode_predicates(self, node):
        column_features = np.zeros(self.column_bucket_count)
        operator_features = np.zeros(self.operator_stat_count)
        literal_features = np.zeros(self.literal_stat_count)

        if node.operator != 'Scan':
            return np.concatenate((column_features, operator_features, literal_features))

        constrained_columns = set()
        num_columns = set()
        date_columns = set()
        str_columns = set()

        predicates = node.data.get('predicates', [])
        if isinstance(predicates, str):
            predicates = []

        for predicate in predicates:
            if not predicate or len(predicate) < 2:
                continue

            column, operator = predicate[0], predicate[1]
            value = predicate[2] if len(predicate) > 2 else None

            if operator == 'raw' or not column:
                continue

            constrained_columns.add(column)

            if operator == '=':
                operator_features[0] += 1
            elif operator == '<':
                operator_features[1] += 1
            elif operator == '>':
                operator_features[2] += 1
            elif operator == 'contains':
                operator_features[3] += 1
            elif operator == 'isnotnull':
                operator_features[4] += 1

            literal_type = self._classify_literal(value)
            if literal_type == 'num':
                literal_features[0] += 1
                num_columns.add(column)
            elif literal_type == 'date':
                literal_features[1] += 1
                date_columns.add(column)
            elif literal_type == 'str':
                literal_features[2] += 1
                str_columns.add(column)

        for column in constrained_columns:
            column_features[self._stable_bucket(column, self.column_bucket_count)] += 1

        literal_features[3] = len(num_columns)
        literal_features[4] = len(date_columns)
        literal_features[5] = len(str_columns)

        return np.concatenate((column_features, operator_features, literal_features))

    def _featurize_not_null_operator(self, node):
        executed = np.ones(1) if node.executed > 0 else np.zeros(1)

        op_onehot = np.zeros(len(self.op_list) + 1)
        if node.operator in self.op_list:
            op_onehot[self.op_list.index(node.operator)] = 1
        if 'Join' in node.operator or "CartesianProduct" in node.operator:
            op_onehot[self.op_list.index('Join')] = 1

        data_info = np.zeros(8)
        if node.operator == 'SortMergeJoin':
            data_info[0] = 1
        if node.operator == 'BroadcastHashJoin':
            data_info[1] = 1
        if node.operator == 'BroadcastNestedLoopJoin':
            data_info[2] = 1
        if node.operator == 'ShuffledHashJoin':
            data_info[3] = 1
        if node.operator == 'CartesianProduct':
            data_info[4] = 1
        if node.operator == 'AQEShuffleRead':
            if node.data['mode'] == 'coalesce':
                data_info[5] = 1
            elif node.data['mode'] == 'local':
                data_info[6] = 1
        if node.operator == 'Exchange':
            data_info[7] = node.data['partition_number']

        tables = np.zeros(len(self.table_list))
        for table in node.tables:
            if table in self.table_list:
                tables[self.table_list.index(table)] = 1

        stats = np.array([node.card, node.size_in_bytes])
        features = [executed, op_onehot, data_info, tables, stats]
        if self.enable_predicate_encoding:
            features.append(self._encode_predicates(node))
        return np.concatenate(features)

    def _featurize_null_operator(self):
        executed = np.zeros(1)
        op_onehot = np.zeros(len(self.op_list) + 1)
        op_onehot[-1] = 1
        data_info = np.zeros(8)
        tables = np.zeros(len(self.table_list))
        stats = np.array([-1, -1])
        features = [executed, op_onehot, data_info, tables, stats]
        if self.enable_predicate_encoding:
            features.append(np.zeros(self.predicate_dim))
        return np.concatenate(features)

    def featurize(self, tree, i=1):
        if len(tree) <= 1:
            return (self._featurize_null_operator(),)
        return self._featurize_not_null_operator(tree[i]), \
            self.featurize(tree, tree[i].lc) if tree[i].lc is not None else (self._featurize_null_operator(),), \
            self.featurize(tree, tree[i].rc) if tree[i].rc is not None else (self._featurize_null_operator(),)
