import torch
import random
from utils.logger import setup_custom_logger
from utils.util import flatten_tree_batch, tree_equal, flatten_tree
from preprocessor.simple_preprocessor import SparkPlanPreprocessor
from models.encoder import UnifiedEncoder
from models.TreeLRUNet import TreeLRUNet
import numpy as np
from config import ServerConfig, TrainConfig
import torch
from torch import optim
from torch import nn
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

logger = setup_custom_logger("TRAIN")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pair_data(path, seed=42):
    logger.info(f"Loading data from {path}")

    data = torch.load(path)

    # 打乱顺序
    random.seed(seed)
    random.shuffle(data)

    # 分离 x 和 y
    pair_x1 = [item[0] for item in data]
    pair_x2 = [item[1] for item in data]
    pair_y = [item[2] for item in data]

    return pair_x1, pair_x2, pair_y

def tree2vector(flattened_tree):
    mat = np.stack(flattened_tree)
    mean_vec = mat.mean(axis=0)
    max_vec = mat.max(axis=0)
    min_vec = mat.min(axis=0)
    return np.concatenate([mean_vec, max_vec, min_vec])

if __name__ == "__main__":
    pair_x1, pair_x2, pair_y = load_pair_data(ServerConfig.data_path)
    log_subdir = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f'/home/hejiahao/Calibra/logs/train_history/{log_subdir}')
    preprocessor = SparkPlanPreprocessor()
    encoder = UnifiedEncoder()

    # 编码训练和验证数据
    pair_x1 = [encoder.featurize(x) for x in pair_x1]
    pair_x2 = [encoder.featurize(x) for x in pair_x2]
    assert len(pair_x1) == len(pair_x2)
    logger.info(f"Number of training pairs: {len(pair_x1)}")

    test = tree_equal(pair_x1[-1], pair_x2[-1])

    filtered_x1, filtered_x2, filtered_y = [], [], []
    removed_count = 0
    for x1, x2, y in zip(pair_x1, pair_x2, pair_y):
        if not tree_equal(x1, x2):
            filtered_x1.append(x1)
            filtered_x2.append(x2)
            filtered_y.append(y)
        else:
            removed_count += 1
    logger.info(f"Removed {removed_count} identical pairs.")
    logger.info(f"Number of remaining training pairs: {len(filtered_x1)}")
    in_features = encoder.in_features
    if len(filtered_x1) > TrainConfig.bootstrap_sample_size:
        vectors = []
        for plan in filtered_x1:
            node_embs = flatten_tree(plan)
            vec = tree2vector(node_embs)
            vectors.append(vec)
        vectors = np.vstack(vectors).astype(np.float32)
        logger.info(f"Shape of vectors: {vectors.shape}")

        mbk = MiniBatchKMeans(n_clusters=TrainConfig.bootstrap_sample_size, max_iter=200, n_init="auto")
        mbk.fit(vectors)
        # 3. Picking representatives
        labels = mbk.labels_
        centers = mbk.cluster_centers_

        selected_indices = []
        for i in range(TrainConfig.bootstrap_sample_size):
            cluster_members = np.where(labels == i)[0]
            if len(cluster_members) == 0:
                continue
            # find plan closest to the centroid
            diffs = vectors[cluster_members] - centers[i]
            # Ensure diffs is 2D for norm calculation
            if diffs.ndim == 1:
                diffs = diffs.reshape(1, -1)
            norms = np.linalg.norm(diffs, axis=1)
            idx = cluster_members[np.argmin(norms)]
            selected_indices.append(idx)

        filtered_x1 = [filtered_x1[i] for i in selected_indices]
        filtered_x2 = [filtered_x2[i] for i in selected_indices]
        filtered_y = [filtered_y[i] for i in selected_indices]
        logger.info(f"Number of remaining training pairs: {len(filtered_x1)}")
        if TrainConfig.save_bootstrap_samples:
            torch.save((filtered_x1, filtered_x2, filtered_y), TrainConfig.bootstrap_samples_save_path)
            logger.info(f"Saved bootstrap samples to {TrainConfig.bootstrap_samples_save_path}")

    # start training
    model = TreeLRUNet(in_features=in_features).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()
    sigmoid = nn.Sigmoid()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"total_params: {total_params}")

    # 将 y 转为 tensor
    filtered_y = torch.tensor(filtered_y, dtype=torch.float32).view(-1, 1).to(device)

    if not TrainConfig.inference_only:
        num_batches = (len(filtered_x1) + TrainConfig.batch_size - 1) // TrainConfig.batch_size

        train_loss_history = []
        val_loss_history = []
        best_val_loss = float('inf')
        for epoch in range(TrainConfig.epochs):
            logger.info(f"Epoch {epoch+1}/{TrainConfig.epochs}")
            # 打乱训练集
            perm = torch.randperm(len(filtered_x1))
            filtered_x1 = [filtered_x1[i] for i in perm]
            filtered_x2 = [filtered_x2[i] for i in perm]
            filtered_y = filtered_y[perm]

            for b in range(num_batches):
                start_idx = b * TrainConfig.batch_size
                end_idx = min((b + 1) * TrainConfig.batch_size, len(filtered_x1))
                batch_x1 = filtered_x1[start_idx:end_idx]
                batch_x2 = filtered_x2[start_idx:end_idx]
                batch_y = filtered_y[start_idx:end_idx]

                flattened_x1 = flatten_tree_batch(batch_x1)
                flattened_x2 = flatten_tree_batch(batch_x2)
                pred_1 = model(flattened_x1)
                pred_2 = model(flattened_x2)

                diff = pred_1 - pred_2
                prob_y = sigmoid(diff)

                loss = loss_fn(prob_y, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                logger.info(f"Batch {b+1}/{num_batches}, Train Loss={loss.item():.2f}")
                train_loss_history.append(loss.item())
                global_step = epoch * num_batches + b + 1  # 全局 step
                writer.add_scalar("Loss/Train", loss.item(), global_step)
        
            torch.save(model.state_dict(), TrainConfig.bs_model_save_path)
            logger.info(f"Saved trained model parameters to {TrainConfig.bs_model_save_path}")
        # ---------------------
        # 绘制训练 loss 曲线
        # ---------------------
        plt.figure(figsize=(8, 5))
        plt.plot(train_loss_history, label="Train Loss")
        plt.xlabel("Iteration")
        plt.ylabel("BCE Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True)
        img_name = TrainConfig.log_save_path + ".png"
        plt.savefig(img_name, dpi=300)
        logger.info(f"Saved training loss curve to {img_name}")
        plt.close()
    else:
        model.load_state_dict(torch.load(TrainConfig.bs_model_save_path))
        logger.info(f"Loaded trained model parameters from {TrainConfig.bs_model_save_path}")
