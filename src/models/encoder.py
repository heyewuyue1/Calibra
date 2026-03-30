from config import EnvironmentConfig
import hashlib
import numpy as np
import re
from utils.logger import setup_custom_logger

logger = setup_custom_logger('ENCODER')

DATE_LITERAL_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
NUMBER_LITERAL_RE = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)")

class OneHotEncoder:
    def __init__(self):
        self.op_list = ['Aggregate', 'Scan', 'Join Inner', 'LogicalQueryStage', 'SortMergeJoin', 'BroadcastHashJoin', 'AQEShuffleRead', 'Exchange', 'BroadcastExchange']
        with open(EnvironmentConfig.table_file) as f:
            self.table_list = [s.strip() for s in f.readlines()]
    
    def __featurize_not_null_operator(self, node):
        arr = np.zeros(len(self.op_list) + 1)
        arr[self.op_list.index(node.operator)] = 1
        tables = np.zeros(len(self.table_list))
        for table in node.tables:
            if table in self.table_list:
                tables[self.table_list.index(table)] = 1
        return np.concatenate((arr, tables, [node.card, node.size_in_bytes]))

    def __featurize_null_operator(self):
        arr = np.zeros(len(self.op_list) + 1)
        arr[-1] = 1  # declare as null vector
        tables = np.zeros(len(self.table_list))
        return np.concatenate((arr, tables, [-1, -1]))

    def featurize(self, tree, i):
        # logger.debug(f'Featurizing node {i} {tree[i].operator}')
        if len(tree) <= 1:
            return self.__featurize_null_operator()
        return self.__featurize_not_null_operator(tree[i]),\
        self.featurize(tree, tree[i].lc) if tree[i].lc is not None else (self.__featurize_null_operator(),),\
        self.featurize(tree, tree[i].rc) if tree[i].rc is not None else (self.__featurize_null_operator(),)


class UnifiedEncoder:
    def __init__(self):
        # executed[1] + op_onehot[7 + 1] + data_info[5]+ table_onehot[table_len] + card[1] + size[1] = 37
        # data_info[5]: SMJ[1]+BHJ[1]+Coalesce[1]+Local[1]+partition_number[1]
        self.op_list = ['Aggregate', 'Scan', 'Join', 'LogicalQueryStage', 'AQEShuffleRead', 'Exchange', 'BroadcastExchange']
        with open(EnvironmentConfig.table_file) as f:
            self.table_list = [s.strip() for s in f.readlines()]
        self.in_features = 1 + 8 + len(self.table_list) + 5 + 2
    
    def __featurize_not_null_operator(self, node):
        executed = np.ones(1) if node.executed > 0 else np.zeros(1)
        
        op_onehot = np.zeros(len(self.op_list) + 1)
        if node.operator in self.op_list:
            op_onehot[self.op_list.index(node.operator)] = 1
        if 'Join' in node.operator:
            op_onehot[self.op_list.index('Join')] = 1
        
        data_info = np.zeros(5)
        if node.operator == 'SortMergeJoin':
            data_info[0] = 1
        if node.operator == 'BroadcastHashJoin':
            data_info[1] = 1
        if node.operator == 'AQEShuffleRead':
            if node.data['mode'] =='coalesce':
                data_info[2] = 1
            elif node.data['mode'] == 'local':
                data_info[3] = 1
        if node.operator == 'Exchange':
            data_info[4] = node.data['partition_number']

        tables = np.zeros(len(self.table_list))
        for table in node.tables:
            if table in self.table_list:
                tables[self.table_list.index(table)] = 1
        
        stats = np.array([node.card, node.size_in_bytes])
        # logger.info(f"{np.concatenate((executed, op_onehot, data_info, tables, stats))}")
        return np.concatenate((executed, op_onehot, data_info, tables, stats))
    
    def __featurize_null_operator(self):
        executed = np.zeros(1)

        op_onehot = np.zeros(len(self.op_list) + 1)
        op_onehot[-1] = 1  # declare as null vector

        data_info = np.zeros(5)
        tables = np.zeros(len(self.table_list))
        stats = np.array([-1, -1])

        return np.concatenate((executed, op_onehot, data_info, tables, stats))

    def featurize(self, tree, i=1):
        # logger.debug(f'Featurizing node {i} {tree[i].operator}')
        if len(tree) <= 1:
            return (self.__featurize_null_operator(),)
        return self.__featurize_not_null_operator(tree[i]),\
        self.featurize(tree, tree[i].lc) if tree[i].lc is not None else (self.__featurize_null_operator(),),\
        self.featurize(tree, tree[i].rc) if tree[i].rc is not None else (self.__featurize_null_operator(),)


class PredicateAwareUnifiedEncoder:
    def __init__(self):
        # executed[1] + op_onehot[7 + 1] + data_info[5] + table_onehot[table_len]
        # + card[1] + size[1] + predicate_features[19]
        self.op_list = ['Aggregate', 'Scan', 'Join', 'LogicalQueryStage', 'AQEShuffleRead', 'Exchange', 'BroadcastExchange']
        with open(EnvironmentConfig.table_file) as f:
            self.table_list = [s.strip() for s in f.readlines()]

        self.column_bucket_count = 8
        self.operator_stat_count = 5
        self.literal_stat_count = 6
        self.predicate_dim = (
            self.column_bucket_count + self.operator_stat_count + self.literal_stat_count
        )
        self.in_features = 1 + 8 + len(self.table_list) + 5 + 2 + self.predicate_dim

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

        for predicate in node.data.get('predicates', []):
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

    def __featurize_not_null_operator(self, node):
        executed = np.ones(1) if node.executed > 0 else np.zeros(1)

        op_onehot = np.zeros(len(self.op_list) + 1)
        if node.operator in self.op_list:
            op_onehot[self.op_list.index(node.operator)] = 1
        if 'Join' in node.operator:
            op_onehot[self.op_list.index('Join')] = 1

        data_info = np.zeros(5)
        if node.operator == 'SortMergeJoin':
            data_info[0] = 1
        if node.operator == 'BroadcastHashJoin':
            data_info[1] = 1
        if node.operator == 'AQEShuffleRead':
            if node.data['mode'] == 'coalesce':
                data_info[2] = 1
            elif node.data['mode'] == 'local':
                data_info[3] = 1
        if node.operator == 'Exchange':
            data_info[4] = node.data['partition_number']

        tables = np.zeros(len(self.table_list))
        for table in node.tables:
            if table in self.table_list:
                tables[self.table_list.index(table)] = 1

        stats = np.array([node.card, node.size_in_bytes])
        predicate_features = self._encode_predicates(node)
        return np.concatenate((executed, op_onehot, data_info, tables, stats, predicate_features))

    def __featurize_null_operator(self):
        executed = np.zeros(1)

        op_onehot = np.zeros(len(self.op_list) + 1)
        op_onehot[-1] = 1

        data_info = np.zeros(5)
        tables = np.zeros(len(self.table_list))
        stats = np.array([-1, -1])
        predicate_features = np.zeros(self.predicate_dim)

        return np.concatenate((executed, op_onehot, data_info, tables, stats, predicate_features))

    def featurize(self, tree, i=1):
        if len(tree) <= 1:
            return (self.__featurize_null_operator(),)
        return self.__featurize_not_null_operator(tree[i]),\
        self.featurize(tree, tree[i].lc) if tree[i].lc is not None else (self.__featurize_null_operator(),),\
        self.featurize(tree, tree[i].rc) if tree[i].rc is not None else (self.__featurize_null_operator(),)
