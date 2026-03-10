import torch
import random
from utils.logger import setup_custom_logger
from utils.util import flatten_tree_batch
from preprocessor.simple_preprocessor import SparkPlanPreprocessor
from models.encoder import UnifiedEncoder
from models.TreeLRUNet import TreeLRUNet
import numpy as np
from config import ServerConfig, TrainConfig
import torch
from torch import optim
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import time
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime

logger = setup_custom_logger("TRAIN")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(path, split_ratio=0.9, actual_split=False, seed=42):
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
    if actual_split:
        return train_x, train_y, val_x, val_y
    else:
        return train_x + val_x, train_y + val_y, val_x, val_y

def eval_qerror(model, data_x, data_y, raw_data, threshold=10):
    """
    评估模型的 Q-error，并打印 q_error > threshold 的样本详细信息。
    
    参数:
        model: 训练好的模型
        data_x: 编码后的输入 (list of trees)
        data_y: 对应的真实值 (torch.Tensor)
        raw_data: 原始数据 (list)，其中每个元素应包含 'plan' 和 'query_stages'
        threshold: 超过该阈值的 Q-error 会被详细输出
    """
    preds = []
    total_t = time.time()
    # Flatten 阶段
    all_flatten_time = 0
    all_infer_time = 0
    all_to_cpu_time = 0
    with torch.no_grad():
        for i in range(0, len(data_x), TrainConfig.batch_size):
            batch_x = data_x[i:i + TrainConfig.batch_size]
            flattened_start = time.time()
            flattened = flatten_tree_batch(batch_x)
            all_flatten_time += time.time() - flattened_start

            infer_start = time.time()
            pred = model(flattened)
            pred = torch.clamp(pred, min=1.0)
            pred = pred.cpu().numpy().flatten()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            all_infer_time += time.time() - infer_start

            to_cpu_start = time.time()
            preds.extend(pred)
            all_to_cpu_time += time.time() - to_cpu_start
    logger.info(f"Flatten time total: {all_flatten_time:.2f}s")
    logger.info(f"Inference time total: {all_infer_time:.2f}s")
    logger.info(f"Tensor→CPU time total: {all_to_cpu_time:.2f}s")
    logger.info(f"Total eval_qerror time: {time.time() - total_t:.2f}s")

    y_true = data_y.cpu().numpy().flatten()
    preds = np.array(preds)

    eps = 1e-6
    y_true = np.maximum(y_true, eps)
    preds = np.maximum(preds, eps)
    q_error = np.maximum(y_true / preds, preds / y_true)

    large_q_indices = np.where(q_error > threshold)[0]
    if len(large_q_indices) == 0:
        logger.info(f"没有找到 q_error > {threshold} 的样本")
    else:
        logger.info(f"找到 {len(large_q_indices)} 个 q_error > {threshold} 的样本:")
        logger.info("-" * 60)
        for i, idx in enumerate(large_q_indices):
            logger.info(f"样本 {i+1} (索引 {idx}):")
            logger.info(f"  y_true: {y_true[idx]}")
            logger.info(f"  y_pred: {preds[idx]}")
            logger.info(f"  q_error: {q_error[idx]:.6f}")

            if idx < len(raw_data):
                sample = raw_data[idx]
                logger.info(f"  plan_info: {sample['plan_info']}")
            logger.info("-" * 40)

    corr = np.corrcoef(y_true, preds)[0, 1]
    return {
        "mean": float(np.mean(q_error)),
        "median": float(np.median(q_error)),
        "p90": float(np.percentile(q_error, 90)),
        "corr": float(corr)
    }

if __name__ == "__main__":
    raw_train_x, train_y, raw_val_x, val_y = load_data(ServerConfig.data_path)
    log_subdir = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f'/home/hejiahao/Calibra/logs/train_history/{log_subdir}')
    preprocessor = SparkPlanPreprocessor()
    encoder = UnifiedEncoder()
    # 遍历所有元素，将 'tree' 重命名为 'plan_info'
    for item in raw_train_x:
        if 'tree' in item and 'plan_info' not in item:
            item['plan_info'] = item.pop('tree')
    # 编码训练和验证数据
    train_plan_x = [encoder.featurize(preprocessor.plan2tree(x["plan_info"])) for x in raw_train_x]
    logger.info(f"Number of training plans: {len(train_plan_x)}")
    val_plan_x = [encoder.featurize(preprocessor.plan2tree(x["plan_info"])) for x in raw_val_x]
    logger.info(f"Number of validation plans: {len(val_plan_x)}")
    
    in_features = len(train_plan_x[0][0])
    logger.info(f"in_features: {in_features}")

    model = TreeLRUNet(in_features=in_features).to(device)
    try:
        model.load_state_dict(torch.load(TrainConfig.bs_model_save_path))
        logger.info(f"Loaded bootstrap model parameters from {TrainConfig.bs_model_save_path}")
    except:
        logger.info(f"No bootstrap model found at {TrainConfig.bs_model_save_path}, training from scratch")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mse_loss_fn = nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"total_params: {total_params}")

    # 将 y 转为 tensor
    train_y = torch.tensor(train_y, dtype=torch.float32).view(-1, 1).to(device)
    val_y = torch.tensor(val_y, dtype=torch.float32).view(-1, 1).to(device)

    if not TrainConfig.inference_only:
        num_batches = (len(train_plan_x) + TrainConfig.batch_size - 1) // TrainConfig.batch_size

        train_loss_history = []
        val_loss_history = []
        best_val_loss = float('inf')
        for epoch in range(TrainConfig.epochs):
            logger.info(f"Epoch {epoch+1}/{TrainConfig.epochs}")
            # 打乱训练集
            perm = torch.randperm(len(train_plan_x))
            train_plan_x = [train_plan_x[i] for i in perm]
            train_y = train_y[perm]

            for b in range(num_batches):
                start_idx = b * TrainConfig.batch_size
                end_idx = min((b + 1) * TrainConfig.batch_size, len(train_plan_x))
                batch_x = train_plan_x[start_idx:end_idx]
                batch_y = train_y[start_idx:end_idx]

                flattened_trees = flatten_tree_batch(batch_x)
                pred = model(flattened_trees)
                mse_loss = mse_loss_fn(pred, batch_y)

                optimizer.zero_grad()
                mse_loss.backward()
                optimizer.step()

                logger.info(f"Batch {b+1}/{num_batches}, Train Loss (MSE) = {mse_loss.item():.2f}")
                train_loss_history.append(mse_loss.item())
                global_step = epoch * num_batches + b + 1  # 全局 step
                writer.add_scalar("Loss/Train(MSE)", mse_loss.item(), global_step)
                 
            # 每个 epoch 结束计算验证集 loss
            with torch.no_grad():
                val_loss_list = []
                val_num_batches = (len(val_plan_x) + TrainConfig.batch_size - 1) // TrainConfig.batch_size
                for vb in range(val_num_batches):
                    start_idx = vb * TrainConfig.batch_size
                    end_idx = min((vb + 1) * TrainConfig.batch_size, len(val_plan_x))
                    batch_val_x = val_plan_x[start_idx:end_idx]
                    batch_val_y = val_y[start_idx:end_idx]

                    flattened_val = flatten_tree_batch(batch_val_x)
                    val_pred = model(flattened_val)
                    val_loss_list.append(mse_loss_fn(val_pred, batch_val_y).item())

                avg_val_loss = sum(val_loss_list) / len(val_loss_list)
                logger.info(f"Epoch {epoch+1} Validation Loss (MSE) = {avg_val_loss:.6f}")
                val_loss_history.append(avg_val_loss)
                writer.add_scalar("Loss/Val(MSE)", avg_val_loss, epoch+1)
                # 早停判断
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    best_model_state = model.state_dict()  # 保存最优参数
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= TrainConfig.patience:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        model.load_state_dict(best_model_state)  # 恢复最优模型
                        break
        
        torch.save(model.state_dict(), TrainConfig.model_save_path)
        logger.info(f"Saved trained model parameters to {TrainConfig.model_save_path}")
        # ---------------------
        # 绘制训练 loss 曲线
        # ---------------------
        plt.figure(figsize=(8, 5))
        plt.plot(train_loss_history, label="Train Loss")
        plt.plot(
            np.linspace(0, len(train_loss_history), len(val_loss_history)),
            val_loss_history,
            label="Val Loss"
        )
        plt.xlabel("Iteration")
        plt.ylabel("MSE Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True)
        img_name = TrainConfig.log_save_path + ".png"
        plt.savefig(img_name, dpi=300)
        logger.info(f"Saved training loss curve to {img_name}")
        plt.close()
    else:
        model.load_state_dict(torch.load(TrainConfig.model_save_path))
        logger.info(f"Loaded trained model parameters from {TrainConfig.model_save_path}")

    # ---------------------
    # 训练结束后计算 Q-error
    # ---------------------

    train_metrics = eval_qerror(model, train_plan_x, train_y, raw_train_x)
    val_metrics = eval_qerror(model, val_plan_x, val_y, raw_val_x)

    # 记录 train Q-error
    writer.add_scalar("QError/Train_Mean", train_metrics["mean"], epoch+1)
    writer.add_scalar("QError/Train_Median", train_metrics["median"], epoch+1)
    writer.add_scalar("QError/Train_90", train_metrics["p90"], epoch+1)
    writer.add_scalar("QError/Train_Corr", train_metrics["corr"], epoch+1)

    # 记录 val Q-error
    writer.add_scalar("QError/Val_Mean", val_metrics["mean"], epoch+1)
    writer.add_scalar("QError/Val_Median", val_metrics["median"], epoch+1)
    writer.add_scalar("QError/Val_90", val_metrics["p90"], epoch+1)
    writer.add_scalar("QError/Val_Corr", val_metrics["corr"], epoch+1)

    df = pd.DataFrame([
        {"dataset": "train", **train_metrics},
        {"dataset": "val", **val_metrics}
    ])
    csv_path = TrainConfig.log_save_path + ".csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved Q-error stats to {csv_path}")
