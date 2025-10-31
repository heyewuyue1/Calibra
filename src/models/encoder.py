from config import EnvironmentConfig
import numpy as np

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
