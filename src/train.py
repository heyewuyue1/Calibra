import torch
import random
from utils.logger import setup_custom_logger
from preprocessor.simple_preprocessor import SparkPlanPreprocessor
from models.encoder import OneHotEncoder
from models.TreeLRUNet import TreeLRUNet
import numpy as np
from config import ServerConfig
import torch
from torch import optim
from torch import nn

logger = setup_custom_logger("TRAIN")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _flatten_tree_batch(trees):
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
        if node is None:
            return None

        curr_idx = len(x)
        value = node[0]
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                pass
        else:
            value = np.array(value)  # 标量处理

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

def load_data(path, split_ratio=0.8, seed=42):
    data = torch.load(path)

    # 打乱顺序
    random.seed(seed)
    random.shuffle(data)

    # 拆分
    split_idx = int(len(data) * split_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # 分离 x 和 y
    train_x = [item["x"] for item in train_data]
    train_y = [item["y"] / 1000_000_000 for item in train_data]
    val_x = [item["x"] for item in val_data]
    val_y = [item["y"] / 1000_000_000 for item in val_data]

    return train_x, train_y, val_x, val_y

if __name__ == "__main__":
    batch_size = 64
    epochs = 5  # 可以自行调整
    train_x, train_y, val_x, val_y = load_data(ServerConfig.data_path)

    preprocessor = SparkPlanPreprocessor()
    encoder = OneHotEncoder()

    # 编码训练和验证数据
    train_x = [encoder.featurize(preprocessor.plan2tree(x["plan"], x["query_stages"]), 1) for x in train_x]
    val_x = [encoder.featurize(preprocessor.plan2tree(x["plan"], x["query_stages"]), 1) for x in val_x]

    in_features = len(train_x[0][0])
    logger.info(f"in_features: {in_features}")

    model = TreeLRUNet(in_features=in_features).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"total_params: {total_params}")

    # 将 y 转为 tensor
    train_y = torch.tensor(train_y, dtype=torch.float32).view(-1, 1).to(device)
    val_y = torch.tensor(val_y, dtype=torch.float32).view(-1, 1).to(device)

    num_batches = (len(train_x) + batch_size - 1) // batch_size

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        # 打乱训练集
        perm = torch.randperm(len(train_x))
        train_x = [train_x[i] for i in perm]
        train_y = train_y[perm]

        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, len(train_x))
            batch_x = train_x[start_idx:end_idx]
            batch_y = train_y[start_idx:end_idx]

            flattened_trees = _flatten_tree_batch(batch_x)
            pred = model(flattened_trees)
            loss = loss_fn(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.info(f"Batch {b+1}/{num_batches}, Train Loss={loss.item():.6f}")

        # 每个 epoch 结束计算验证集 loss
        with torch.no_grad():
            val_loss_list = []
            val_num_batches = (len(val_x) + batch_size - 1) // batch_size
            for vb in range(val_num_batches):
                start_idx = vb * batch_size
                end_idx = min((vb + 1) * batch_size, len(val_x))
                batch_val_x = val_x[start_idx:end_idx]
                batch_val_y = val_y[start_idx:end_idx]

                flattened_val = _flatten_tree_batch(batch_val_x)
                val_pred = model(flattened_val)
                val_loss_list.append(loss_fn(val_pred, batch_val_y).item())

            avg_val_loss = sum(val_loss_list) / len(val_loss_list)
            logger.info(f"Epoch {epoch+1} Validation Loss={avg_val_loss:.6f}")



# if __name__ == "__main__":
#     train_x, train_y, val_x, val_y = load_data(ServerConfig.data_path)
#     for x, y in zip(train_x[:3], train_y[:3]):
#         logger.info(f"x: {str(x)[:100]}, y: {y}")
#     # {"query_id": request.sessionName, "plan_type": plan_type, "plan": plan, "query_stages": query_stages}
#     preprocessor = SparkPlanPreprocessor()
#     encoder = OneHotEncoder()
#     # tarin_x = [{"query_id": x["query_id"], "plan_type": x["plan_type"], "plan": encoder.__featurize(preprocessor.plan2tree(x["plan"], x["query_stages"], 1))} for x in train_x]
#     # val_x = [{"query_id": x["query_id"], "plan_type": x["plan_type"], "plan": encoder.__featurize(preprocessor.plan2tree(x["plan"], x["query_stages"], 1))} for x in val_x]
#     train_x = [encoder.featurize(preprocessor.plan2tree(x["plan"], x["query_stages"]), 1) for x in train_x][:1]
#     val_x = [encoder.featurize(preprocessor.plan2tree(x["plan"], x["query_stages"]), 1) for x in val_x]
#     logger.info(f"in_features: {len(train_x[0][0])}")
#     model = TreeLRUNet(in_features=len(train_x[0][0])).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#     loss_fn = nn.MSELoss()
#     target = torch.tensor(train_y[:1], dtype=torch.float32).view(-1, 1).to(device)

#     total_params = sum(p.numel() for p in model.parameters())
#     logger.info(f"total_params: {total_params}")

#     for step in range(200):
#         flattened_trees = _flatten_tree_batch(train_x)
#         pred = model(flattened_trees)
#         loss = loss_fn(pred, target)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if (step + 1) % 50 == 0:
#             logger.info(f"Step {step+1}, Loss={loss.item():.6f}")

#     # 测试输出
#     out_test = model(flattened_trees)
#     logger.info(f"TreeLRUNet 输出: {out_test[0][0].item()}, GT: {target[0][0].item()}")
