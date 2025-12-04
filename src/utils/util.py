import hashlib
import operator
import numpy as np
import torch
from functools import reduce
from utils.logger import setup_custom_logger

logger = setup_custom_logger("UTIL")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def hash_query_plan(plan: str):
    """Generate a hash fingerprint for the result retrieved from the connector to assert that results are (probably) identical.
    Its important to round floats here, e.g. using 2 decimal places."""
    flattened_result = reduce(operator.concat, plan)
    normalized_result = tuple(map(lambda item: round(item, 2) if isinstance(item, float) else item, flattened_result))
    sha256 = hashlib.sha256()
    for item in normalized_result:
        sha256.update(str(item).encode())
    return sha256.hexdigest()

def flatten_tree_batch(trees):
    """
    批量先序遍历多棵树，将节点值平铺后 zero-padding 对齐。
    
    参数:
        trees: list，每个元素是形如 [value, left_subtree, right_subtree] 的嵌套列表
               value 可以是标量或 1D np.array

    返回:
        x_padded: np.ndarray, shape = (batch, max_nodes, feature_dim)
        idx_list: list，每棵树的 (left_idx, right_idx) 列表
    """
    x_list = []
    idx_list = []

    def recur(node, x, idx):
        curr_idx = len(x)
        value = node[0]
        x.append(value)
        idx.append((None, None))  # 占位

        left_idx, right_idx = None, None
        if len(node) > 1:
            left_idx = recur(node[1], x, idx)
        if len(node) > 2:
            right_idx = recur(node[2], x, idx)

        idx[curr_idx] = (left_idx, right_idx)
        return curr_idx

    # flatten 每棵树
    for tree in trees:
        x, idx = [], []
        recur(tree, x, idx)
        x_list.append(np.stack(x))  # shape = (num_nodes, feat_dim)
        idx_list.append(idx)

    # zero-padding
    max_nodes = max(arr.shape[0] for arr in x_list)
    feat_dim = x_list[0].shape[1]

    x_padded = np.zeros((len(x_list), max_nodes, feat_dim), dtype=np.float32)
    for i, arr in enumerate(x_list):
        x_padded[i, :arr.shape[0], :] = arr

    return torch.from_numpy(x_padded).to(device), idx_list

def tree_equal(tree1, tree2):
    # 两者都是 ndarray，直接比较
    if isinstance(tree1, np.ndarray) and isinstance(tree2, np.ndarray):
        return np.array_equal(tree1, tree2)
    
    # 两者都是 tuple（内部含：特征, 左, 右）
    if isinstance(tree1, tuple) and isinstance(tree2, tuple):
        if len(tree1) != len(tree2):
            return False
        # 正常三元结构 (feature, left, right)
        if len(tree1) == 3:
            return (tree_equal(tree1[0], tree2[0]) and 
                    tree_equal(tree1[1], tree2[1]) and 
                    tree_equal(tree1[2], tree2[2]))
        # 叶子节点结构 (np.ndarray,)
        elif len(tree1) == 1:
            return tree_equal(tree1[0], tree2[0])
        else:
            return False
    
    # 其他情况（比如 None 或标量）
    return tree1 == tree2

def flatten_tree(tree):
    x = []
    def recur(node):
        value = tree[0]
        assert isinstance(value, np.ndarray)
        x.append(value)
        if len(node) > 1:
            recur(node[1])
        if len(node) > 2:
            recur(node[2])
    recur(tree)
    
    # return List[np.array]
    return x