from utils.logger import setup_custom_logger
from preprocessor.node import Node
from config import EnvironmentConfig
logger = setup_custom_logger("PREPROCESSOR")

class SparkPlanPreprocessor:

    def __init__(self) -> None:
        self.op_list = ['ShuffleExchangeExec', 'SortExec', 'FilterExec', 'ProjectExec', 'LocalTableScanExec', 'BroadcastQueryStageExec', 'BroadcastHashJoinExec', 'SortMergeJoinExec', 'SortAggregateExec', 'ShuffleQueryStageExec', 'FileSourceScanExec', 'BroadcastExchangeExec']
        self.op_dim = len(self.op_list)
        self.table_list = []
        with open(EnvironmentConfig.table_file) as f:
            self.table_list = [s.strip() for s in f.readlines()]
        self.table_dim = len(self.table_list)
        logger.info(f"Got {self.table_dim} tables: {self.table_list}")
        self.feature_dim = self.op_dim + self.table_dim
        logger.info(f"feature dim:{self.feature_dim}")

    def build_tree(self, nodes):
        def helper(it):
            node_info = next(it)
            op = node_info["class"].split(".")[-1]
            num_children = node_info["num-children"]
            cur_node = Node(operator=op)
            if num_children >= 1:
                cur_node.lc = helper(it)
            if num_children == 2:
                cur_node.rc = helper(it)

            return cur_node
    
        return helper(iter(nodes))
    
    # --- 树形可视化（带 / \ ）
    def print_tree_ascii(self,node, indent="", is_left=True):
        if node.rc:
            new_indent = indent + ("│   " if is_left else "    ")
            self.print_tree_ascii(node.rc, new_indent, False)

        logger.info(indent + ("└── " if is_left else "┌── ") + node.operator)

        if node.lc:
            new_indent = indent + ("    " if is_left else "│   ")
            self.print_tree_ascii(node.lc, new_indent, True)
