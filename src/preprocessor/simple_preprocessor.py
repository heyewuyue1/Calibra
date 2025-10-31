from utils.logger import setup_custom_logger
from preprocessor.node import Node
from config import EnvironmentConfig
import numpy as np
import re

logger = setup_custom_logger("PREPROCESSOR")

class SparkPlanPreprocessor:

    def __init__(self) -> None:
        self.op_list = ['Aggregate', 'Scan', 'Join Inner', 'LogicalQueryStage', 'SortMergeJoin', 'BroadcastHashJoin', 'AQEShuffleRead', 'Exchange', 'BroadcastExchange']
        self.op_dim = len(self.op_list)
        self.table_list = []
        with open(EnvironmentConfig.table_file) as f:
            self.table_list = [s.strip() for s in f.readlines()]
        self.table_dim = len(self.table_list)
        logger.info(f"Got {self.table_dim} tables: {self.table_list}")
        self.feature_dim = self.op_dim + self.table_dim
        logger.info(f"feature dim:{self.feature_dim}")

    def tranverse_fix_join_tables(self, tree, i):
        if tree[i].lc is not None:
            tree[i].tables = list(set(tree[i].tables + self.tranverse_fix_join_tables(tree, tree[i].lc)))
        if tree[i].rc is not None:
            tree[i].tables = list(set(tree[i].tables + self.tranverse_fix_join_tables(tree, tree[i].rc)))
        if 'Join' in tree[i].operator:
            logger.debug(f'Fixed join table: {tree[i].tables}')
        return tree[i].tables

    def query_stage2tree(self, tree, i, executed, stage_plan):
        lines = stage_plan.split('\n')
        colon = 0
        join_stack = []
        prev_idx = i - 1
        for j in range(len(lines)):
            if lines[j].strip() == '':
                continue
            tables = []
            data = {}
            if 'BroadcastExchange' in lines[j]:
                op = 'BroadcastExchange'
            elif 'Exchange' in lines[j]:
                op = 'Exchange'
                data = {
                    "key": lines[j].split('(')[-1].split(')')[0].split(', ')[0].split('#')[0],
                    "partition_number": int(re.search(r"hashpartitioning\s*\([^)]*,\s*(\d+)\s*\)", lines[j]).group(1)) if re.search(r"hashpartitioning\s*\([^)]*,\s*(\d+)\s*\)", lines[j]) else None
                }
            elif 'SortMergeJoin' in lines[j]:
                op = 'SortMergeJoin'
                pattern = r"SortMergeJoin \[([^\]]+)\], \[([^\]]+)\]"
                match = re.search(pattern, lines[j])
                left_keys = re.findall(r"(\w+)#\d+", match.group(1))
                right_keys = re.findall(r"(\w+)#\d+", match.group(2))
                data = {
                    "left": list(dict.fromkeys(left_keys))[0] if left_keys else None,
                    "right": list(dict.fromkeys(right_keys))[0] if right_keys else None
                }
                join_stack.append(len(tree))
            elif 'BroadcastHashJoin' in lines[j]:
                op = 'BroadcastHashJoin'
                pattern = r"\b\w+Join\s*\[([^\]]+)\],\s*\[([^\]]+)\].*?\b(BuildRight|BuildLeft)\b"
                match = re.search(pattern, lines[j])
                left_keys = re.findall(r"(\w+)#\d+", match.group(1))
                right_keys = re.findall(r"(\w+)#\d+", match.group(2))
                build_side = match.group(3)

                data = {
                    "left": list(dict.fromkeys(left_keys))[0] if left_keys else None,
                    "right": list(dict.fromkeys(right_keys))[0] if right_keys else None,
                    "build_side": build_side
                }
                join_stack.append(len(tree))
            elif 'AQEShuffleRead' in lines[j]:
                op = 'AQEShuffleRead'
                data = {
                    'mode': lines[j].split('AQEShuffleRead ')[-1]
                }
            elif 'FileScan' in lines[j]:
                op = 'Scan'
                # 提取表名
                table_pattern = r"spark_catalog\.[\w\.]+\.([\w_]+)\["
                table_match = re.search(table_pattern, lines[j])
                table = table_match.group(1) if table_match else None
                tables = [table]

                # 提取 columns（方括号内内容）
                cols_pattern = r"\[([^\]]+)\]"
                cols_match = re.search(cols_pattern, lines[j])
                columns = re.findall(r"(\w+)#?\d*", cols_match.group(1)) if cols_match else []

                # 提取 DataFilters 中的谓词
                filters_pattern = r"DataFilters:\s*\[([^\]]*)\]"
                filters_match = re.search(filters_pattern, lines[j])
                predicates = re.findall(r"([a-zA-Z_]+\([^)]*\))", filters_match.group(1)) if filters_match else []

                # 去掉 #数字
                predicates = [re.sub(r"#\d+", "", p) for p in predicates]
                columns = list(dict.fromkeys(columns))  # 去重保序

                data = {
                    "columns": columns,
                    "predicates": predicates
                }
            
            else:
                continue
            tree.append(Node(op, executed, tables, -1, -1, data))
            logger.debug(f"Append Node:\n{op} {executed} {tables} -1 -1")
            if colon <= lines[j].count(':'):
                tree[prev_idx].lc = len(tree) - 1
            else:
                tree[join_stack[-1]].rc = len(tree) - 1
                join_stack.pop()
            colon = lines[j].count(':')
            prev_idx = len(tree) - 1
        return tree

    def plan2tree(self, plan, executed_row_counts):
        lines = plan.split('\n')
        tree = []
        # logger.info(executed_row_counts)
        
        colon = 0  # count the colons to identify the depth of the tree
        join_stack = []
        tree.append(Node('Root', 0, [], -1, -1))
        prev_idx = 0
        
        for i in range(len(lines)):
            if lines[i].strip() == '':
                continue
            tables = []
            data = {}
            executed = 0
            card = size_in_bytes = -1
            if 'Aggregate' in lines[i]:
                op = 'Aggregate'
                # Maybe add AggExpr
            elif 'Relation' in lines[i]:
                op = 'Scan'
                if 'HiveTableRelation' in lines[i]:
                    tables = [lines[i].split('`')[-2]]
                else:
                    tables = [lines[i].split('[')[0].split('.')[-1]]
                columns = []
                predicates = ""
                if 'Project' in lines[i - 2]:
                    columns = lines[i - 2].strip().split('Project [')[-1][:-1].split(', ')
                    for k in range(len(columns)):
                        columns[k] = columns[k].split('#')[0]
                if 'Filter' in lines[i - 1]:
                    predicates = lines[i - 1].strip().split('Filter ')[-1]
                data = {
                    "columns": columns,
                    "predicates": predicates
                }          
            elif 'Join Inner' in lines[i]:
                op = 'Join Inner'
                join_stack.append(len(tree))
            elif 'LogicalQueryStage' in lines[i]:
                op = 'LogicalQueryStage'
                stage = lines[i].split('- ')[-1].strip()
                executed = executed_row_counts[stage][0]
                card = np.log1p(eval(executed_row_counts[stage][1]))
                size_in_bytes = np.log1p(eval(executed_row_counts[stage][2]))
            else:
                continue
            tree.append(Node(op, executed, tables, card, size_in_bytes, data))
            cur_loc = len(tree) - 1
            if op == 'LogicalQueryStage':
                tree = self.query_stage2tree(tree, len(tree), executed, executed_row_counts[stage][3])
            if colon <= lines[i].count(':'):
                tree[prev_idx].lc = cur_loc
            else:
                tree[join_stack[-1]].rc = cur_loc
                join_stack.pop()
            colon = lines[i].count(':')
            prev_idx = len(tree) - 1
        self.tranverse_fix_join_tables(tree, 0)
        return tree
    
    def print_tree(self, tree, i, depth=0):
        if i is None:
            return
        indent = "    " * depth
        logger.info(
            f"{indent}{tree[i].operator} {tree[i].executed} "
            f"{tree[i].tables} {tree[i].data} "
            f"{tree[i].card} {tree[i].size_in_bytes}"
        )
        self.print_tree(tree, tree[i].lc, depth + 1)
        self.print_tree(tree, tree[i].rc, depth + 1)
