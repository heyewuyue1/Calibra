import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from torch import nn, optim
from torch.utils.tensorboard.writer import SummaryWriter

from config import (
    DEFAULT_BENCHMARK,
    TrainConfig,
    ensure_dir,
    ensure_parent_dir,
    get_run_artifacts,
    update_manifest,
)
from models.TreeLRUNet import TreeLRUNet
from models.encoder import UnifiedFeatureEncoder
from utils.logger import setup_custom_logger
from utils.util import flatten_tree_batch_for_tree_lru, flatten_tree, tree_equal

logger = setup_custom_logger("TRAIN")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--run-id")
    parser.add_argument("--pair-data-path")
    parser.add_argument("--model-save-path")
    parser.add_argument("--tensorboard-dir")
    parser.add_argument("--loss-plot-path")
    parser.add_argument("--bootstrap-sample-path")
    parser.add_argument("--bootstrap-sample-size", type=int, default=TrainConfig.bootstrap_sample_size)
    parser.add_argument(
        "--predicate-encoding",
        dest="predicate_encoding",
        action="store_true",
    )
    parser.add_argument(
        "--no-predicate-encoding",
        dest="predicate_encoding",
        action="store_false",
    )
    parser.set_defaults(predicate_encoding=TrainConfig.enable_predicate_encoding)
    return parser.parse_args()


def load_pair_data(path, seed=42):
    logger.info("Loading data from %s", path)
    data = torch.load(path)
    random = __import__("random")
    random.seed(seed)
    random.shuffle(data)
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
    args = parse_args()
    artifacts = get_run_artifacts(
        benchmark=args.benchmark,
        run_id=args.run_id,
        predicate_encoding=args.predicate_encoding,
    )
    TrainConfig.current_time = artifacts.run_id
    TrainConfig.enable_predicate_encoding = args.predicate_encoding

    pair_data_path = args.pair_data_path or artifacts.bootstrap_data_path
    model_save_path = args.model_save_path or artifacts.bootstrap_model_path
    tensorboard_dir = args.tensorboard_dir or artifacts.bootstrap_tensorboard_dir
    loss_plot_path = args.loss_plot_path or artifacts.bootstrap_loss_plot_path
    bootstrap_sample_path = args.bootstrap_sample_path or artifacts.bootstrap_samples_path

    ensure_parent_dir(model_save_path)
    ensure_parent_dir(loss_plot_path)
    ensure_parent_dir(bootstrap_sample_path)
    ensure_dir(tensorboard_dir)

    update_manifest(
        artifacts.manifest_path,
        {
            **artifacts.manifest_defaults(),
            "bootstrap_train": {
                "loss_plot_path": loss_plot_path,
                "bootstrap_sample_size": args.bootstrap_sample_size,
                "predicate_encoding": args.predicate_encoding,
            },
        },
    )

    pair_x1, pair_x2, pair_y = load_pair_data(pair_data_path)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    encoder = UnifiedFeatureEncoder(enable_predicate_encoding=args.predicate_encoding)

    pair_x1 = [encoder.featurize(x) for x in pair_x1]
    pair_x2 = [encoder.featurize(x) for x in pair_x2]
    assert len(pair_x1) == len(pair_x2)
    logger.info("Number of training pairs: %d", len(pair_x1))
    logger.info("Predicate encoding enabled: %s", args.predicate_encoding)

    filtered_x1, filtered_x2, filtered_y = [], [], []
    removed_count = 0
    for x1, x2, y in zip(pair_x1, pair_x2, pair_y):
        if not tree_equal(x1, x2):
            filtered_x1.append(x1)
            filtered_x2.append(x2)
            filtered_y.append(y)
        else:
            removed_count += 1
    logger.info("Removed %d identical pairs.", removed_count)
    logger.info("Number of remaining training pairs: %d", len(filtered_x1))

    in_features = encoder.in_features
    sample_size = args.bootstrap_sample_size
    if sample_size > 0 and len(filtered_x1) > sample_size:
        vectors = []
        for plan in filtered_x1:
            node_embs = flatten_tree(plan)
            vec = tree2vector(node_embs)
            vectors.append(vec)
        vectors = np.vstack(vectors).astype(np.float32)
        logger.info("Shape of vectors: %s", vectors.shape)

        mbk = MiniBatchKMeans(n_clusters=sample_size, max_iter=200, n_init="auto")
        mbk.fit(vectors)
        labels = mbk.labels_
        centers = mbk.cluster_centers_

        selected_indices = []
        for i in range(sample_size):
            cluster_members = np.where(labels == i)[0]
            if len(cluster_members) == 0:
                continue
            diffs = vectors[cluster_members] - centers[i]
            if diffs.ndim == 1:
                diffs = diffs.reshape(1, -1)
            norms = np.linalg.norm(diffs, axis=1)
            idx = cluster_members[np.argmin(norms)]
            selected_indices.append(idx)

        filtered_x1 = [filtered_x1[i] for i in selected_indices]
        filtered_x2 = [filtered_x2[i] for i in selected_indices]
        filtered_y = [filtered_y[i] for i in selected_indices]
        logger.info("Number of remaining training pairs after clustering: %d", len(filtered_x1))
        if TrainConfig.save_bootstrap_samples:
            torch.save((filtered_x1, filtered_x2, filtered_y), bootstrap_sample_path)
            logger.info("Saved bootstrap samples to %s", bootstrap_sample_path)

    model = TreeLRUNet(in_features=in_features).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()
    sigmoid = nn.Sigmoid()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("total_params: %d", total_params)

    filtered_y = torch.tensor(filtered_y, dtype=torch.float32).view(-1, 1).to(device)

    if not TrainConfig.inference_only:
        num_batches = (len(filtered_x1) + TrainConfig.batch_size - 1) // TrainConfig.batch_size
        train_loss_history = []

        for epoch in range(TrainConfig.epochs):
            logger.info("Epoch %d/%d", epoch + 1, TrainConfig.epochs)
            perm = torch.randperm(len(filtered_x1))
            filtered_x1 = [filtered_x1[i] for i in perm]
            filtered_x2 = [filtered_x2[i] for i in perm]
            filtered_y = filtered_y[perm]

            for batch_idx in range(num_batches):
                start_idx = batch_idx * TrainConfig.batch_size
                end_idx = min((batch_idx + 1) * TrainConfig.batch_size, len(filtered_x1))
                batch_x1 = filtered_x1[start_idx:end_idx]
                batch_x2 = filtered_x2[start_idx:end_idx]
                batch_y = filtered_y[start_idx:end_idx]

                flattened_x1 = flatten_tree_batch_for_tree_lru(batch_x1)
                flattened_x2 = flatten_tree_batch_for_tree_lru(batch_x2)
                pred_1 = model(flattened_x1)
                pred_2 = model(flattened_x2)

                diff = pred_1 - pred_2
                prob_y = sigmoid(diff)
                loss = loss_fn(prob_y, batch_y)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                logger.info("Batch %d/%d, Train Loss=%.4f", batch_idx + 1, num_batches, loss.item())
                train_loss_history.append(loss.item())
                global_step = epoch * num_batches + batch_idx + 1
                writer.add_scalar("Loss/Train(BCE)", loss.item(), global_step)

            torch.save(model.state_dict(), model_save_path)
            logger.info("Saved trained model parameters to %s", model_save_path)

        plt.figure(figsize=(8, 5))
        plt.plot(train_loss_history, label="Train Loss")
        plt.xlabel("Iteration")
        plt.ylabel("BCE Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(loss_plot_path, dpi=300)
        logger.info("Saved training loss curve to %s", loss_plot_path)
        plt.close()
    else:
        model.load_state_dict(torch.load(model_save_path))
        logger.info("Loaded trained model parameters from %s", model_save_path)

    writer.close()
    update_manifest(
        artifacts.manifest_path,
        {
            "bootstrap_train": {
                "model_path": model_save_path,
                "loss_plot_path": loss_plot_path,
                "tensorboard_dir": tensorboard_dir,
            },
        },
    )
