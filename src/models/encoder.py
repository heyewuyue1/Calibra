from config import EnvironmentConfig
import numpy as np
from utils.logger import setup_custom_logger

logger = setup_custom_logger('ENCODER')

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
