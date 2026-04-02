import copy
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.tensorboard.writer import SummaryWriter

from config import ServerConfig, TrainConfig
from models.TreeLRUNet import TreeLRUNet
from models.encoder import UnifiedFeatureEncoder
from preprocessor.sparkplanpreprocessor import SparkPlanPreprocessor
from utils.logger import setup_custom_logger
from utils.util import flatten_tree_batch_for_tree_lru

logger = setup_custom_logger("TRAIN")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(path, split_ratio=0.9, seed=42):
    data = torch.load(path)
    random.seed(seed)
    random.shuffle(data)
    split_idx = int(len(data) * split_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    train_x = [normalize_record(item["x"]) for item in train_data]
    val_x = [normalize_record(item["x"]) for item in val_data]
    train_y = [item["y"] / 1_000_000_000 for item in train_data]
    val_y = [item["y"] / 1_000_000_000 for item in val_data]
    return train_x, train_y, val_x, val_y


def normalize_record(record):
    if "plan_info" in record:
        return record
    if "tree" in record:
        normalized = dict(record)
        normalized["plan_info"] = normalized.pop("tree")
        return normalized
    raise KeyError("Expected plan_info or tree in record")


def load_dataset():
    default_path = ServerConfig.data_path
    try:
        sampled_path = default_path.replace('.pt', 's.pt')
        train_x, train_y, val_x, val_y = load_data(sampled_path)
        logger.info("Loaded training samples from %s", sampled_path)
        return train_x, train_y, val_x, val_y
    except Exception:
        train_x, train_y, val_x, val_y = load_data(default_path)
        logger.info("Loaded training samples from %s", default_path)
        return train_x, train_y, val_x, val_y


def encode_plans(records, encoder, preprocessor):
    return [encoder.featurize(preprocessor.plan2tree(record["plan_info"])) for record in records]


def evaluate_loss(model, plans, labels, batch_size, loss_fn):
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(plans), batch_size):
            batch_plans = plans[start:start + batch_size]
            batch_labels = labels[start:start + batch_size]
            batch = flatten_tree_batch_for_tree_lru(batch_plans)
            pred = model(batch)
            losses.append(loss_fn(pred, batch_labels).item())
    return sum(losses) / len(losses)


def evaluate_qerror(model, plans, labels, batch_size):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(plans), batch_size):
            batch_plans = plans[start:start + batch_size]
            batch = flatten_tree_batch_for_tree_lru(batch_plans)
            pred = model(batch)
            pred = torch.clamp(pred, min=1e-6)
            preds.extend(pred.detach().cpu().numpy().flatten())

    y_true = labels.detach().cpu().numpy().flatten()
    preds = np.array(preds)
    y_true = np.maximum(y_true, 1e-6)
    preds = np.maximum(preds, 1e-6)
    q_error = np.maximum(y_true / preds, preds / y_true)
    corr = float(np.corrcoef(y_true, preds)[0, 1]) if len(y_true) > 1 else 0.0
    return {
        "mean": float(np.mean(q_error)),
        "median": float(np.median(q_error)),
        "p90": float(np.percentile(q_error, 90)),
        "corr": corr,
    }


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    raw_train_x, train_y, raw_val_x, val_y = load_dataset()
    log_subdir = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f'/home/hejiahao/Calibra/logs/train_history/{log_subdir}')

    preprocessor = SparkPlanPreprocessor()
    encoder = UnifiedFeatureEncoder(
        enable_predicate_encoding=TrainConfig.enable_predicate_encoding
    )
    train_plans = encode_plans(raw_train_x, encoder, preprocessor)
    val_plans = encode_plans(raw_val_x, encoder, preprocessor)

    in_features = len(train_plans[0][0])
    logger.info("Number of training plans: %d", len(train_plans))
    logger.info("Number of validation plans: %d", len(val_plans))
    logger.info("Predicate encoding enabled: %s", TrainConfig.enable_predicate_encoding)
    logger.info("in_features: %d", in_features)

    model = TreeLRUNet(in_features=in_features).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    train_y = torch.tensor(train_y, dtype=torch.float32).view(-1, 1).to(device)
    val_y = torch.tensor(val_y, dtype=torch.float32).view(-1, 1).to(device)

    best_val_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    num_batches = (len(train_plans) + TrainConfig.batch_size - 1) // TrainConfig.batch_size

    if not TrainConfig.inference_only:
        train_loss_history = []
        val_loss_history = []
        epochs_no_improve = 0

        for epoch in range(TrainConfig.epochs):
            model.train()
            perm = torch.randperm(len(train_plans)).tolist()
            shuffled_plans = [train_plans[i] for i in perm]
            shuffled_y = train_y[perm]
            epoch_losses = []

            for batch_idx in range(num_batches):
                start = batch_idx * TrainConfig.batch_size
                end = min((batch_idx + 1) * TrainConfig.batch_size, len(shuffled_plans))
                batch_plans = shuffled_plans[start:end]
                batch_y = shuffled_y[start:end]

                batch = flatten_tree_batch_for_tree_lru(batch_plans)
                pred = model(batch)
                loss = loss_fn(pred, batch_y)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
                train_loss_history.append(loss.item())
                global_step = epoch * num_batches + batch_idx + 1
                writer.add_scalar("Loss/Train(MSE)", loss.item(), global_step)
                logger.info(
                    "Epoch %d/%d Batch %d/%d Train Loss (MSE) = %.6f",
                    epoch + 1,
                    TrainConfig.epochs,
                    batch_idx + 1,
                    num_batches,
                    loss.item(),
                )

            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            avg_val_loss = evaluate_loss(model, val_plans, val_y, TrainConfig.batch_size, loss_fn)
            val_loss_history.append(avg_val_loss)
            writer.add_scalar("Loss/Val(MSE)", avg_val_loss, epoch + 1)
            logger.info(
                "Epoch %d/%d train_mse=%.6f val_mse=%.6f",
                epoch + 1,
                TrainConfig.epochs,
                avg_train_loss,
                avg_val_loss,
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= TrainConfig.patience:
                    logger.info("Early stopping triggered at epoch %d", epoch + 1)
                    break

        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), TrainConfig.model_save_path())
        logger.info("Saved trained model parameters to %s", TrainConfig.model_save_path())
    else:
        model.load_state_dict(torch.load(TrainConfig.model_save_path()))
        logger.info("Loaded trained model parameters from %s", TrainConfig.model_save_path())

    train_metrics = evaluate_qerror(model, train_plans, train_y, TrainConfig.batch_size)
    val_metrics = evaluate_qerror(model, val_plans, val_y, TrainConfig.batch_size)

    writer.add_scalar("QError/Train_Mean", train_metrics["mean"], 1)
    writer.add_scalar("QError/Train_Median", train_metrics["median"], 1)
    writer.add_scalar("QError/Train_90", train_metrics["p90"], 1)
    writer.add_scalar("QError/Train_Corr", train_metrics["corr"], 1)
    writer.add_scalar("QError/Val_Mean", val_metrics["mean"], 1)
    writer.add_scalar("QError/Val_Median", val_metrics["median"], 1)
    writer.add_scalar("QError/Val_90", val_metrics["p90"], 1)
    writer.add_scalar("QError/Val_Corr", val_metrics["corr"], 1)

    df = pd.DataFrame([
        {"dataset": "train", **train_metrics},
        {"dataset": "val", **val_metrics}
    ])
    csv_path = TrainConfig.log_save_path() + ".csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved Q-error stats to %s", csv_path)
