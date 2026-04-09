import argparse
from collections import Counter

import torch
import uvicorn
from fastapi import FastAPI

from config import (
    DEFAULT_BENCHMARK,
    EnvironmentConfig,
    TrainConfig,
    get_pretrained_run_spec,
    get_run_artifacts,
    update_manifest,
)
from models.TreeLRUNet import TreeLRUNet
from models.encoder import UnifiedFeatureEncoder
from preprocessor.sparkplanpreprocessor import SparkPlanPreprocessor
from request_models import CostRequest, CostResponse
from utils.logger import setup_custom_logger
from utils.util import flatten_tree_batch_for_tree_lru


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark")
    parser.add_argument("--database")
    parser.add_argument("--table-key")
    parser.add_argument("--model-path")
    parser.add_argument("--run-id")
    parser.add_argument("--port", type=int, default=10533)
    parser.add_argument("--predicate-encoding", dest="predicate_encoding", action="store_true")
    parser.add_argument("--no-predicate-encoding", dest="predicate_encoding", action="store_false")
    parser.add_argument(
        "--cost-shaping-profile",
        choices=["none", "stack_broadcast"],
        default="none",
    )
    parser.set_defaults(predicate_encoding=TrainConfig.enable_predicate_encoding)
    return parser.parse_args()


args = parse_args()
logger = setup_custom_logger("SERVER")
selected_benchmark = args.benchmark or DEFAULT_BENCHMARK
artifacts = get_run_artifacts(
    benchmark=selected_benchmark,
    run_id=args.run_id,
    predicate_encoding=args.predicate_encoding,
)
TrainConfig.current_time = artifacts.run_id
TrainConfig.enable_predicate_encoding = args.predicate_encoding

if args.benchmark and args.run_id is None:
    spec = get_pretrained_run_spec(args.benchmark)
    EnvironmentConfig.configure(
        database=args.database or spec.database,
        table_key=args.table_key or spec.table_key,
    )
    model_path = args.model_path or spec.model_path
else:
    if args.benchmark:
        spec = get_pretrained_run_spec(args.benchmark)
        EnvironmentConfig.configure(
            database=args.database or spec.database,
            table_key=args.table_key or spec.table_key,
        )
    elif args.database or args.table_key:
        EnvironmentConfig.configure(
            database=args.database or EnvironmentConfig.database,
            table_key=args.table_key or EnvironmentConfig.table_key,
        )
    model_path = args.model_path or artifacts.model_path

preprocessor = SparkPlanPreprocessor()

MIN_PREDICTION_SECONDS = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = UnifiedFeatureEncoder(enable_predicate_encoding=args.predicate_encoding)
model = TreeLRUNet(in_features=encoder.in_features).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

app = FastAPI()
logger.info("Loaded model from %s", model_path)
logger.info(
    "Runtime environment: benchmark=%s database=%s table_key=%s predicate_encoding=%s in_features=%s cost_shaping_profile=%s",
    args.benchmark,
    EnvironmentConfig.database,
    EnvironmentConfig.table_key,
    args.predicate_encoding,
    encoder.in_features,
    args.cost_shaping_profile,
)

update_manifest(
    artifacts.manifest_path,
    {
        **artifacts.manifest_defaults(),
        "server": {
            "benchmark": selected_benchmark,
            "database": EnvironmentConfig.database,
            "table_key": EnvironmentConfig.table_key,
            "run_id": artifacts.run_id,
            "model_path": model_path,
            "predicate_encoding": args.predicate_encoding,
            "cost_shaping_profile": args.cost_shaping_profile,
            "port": args.port,
            "log_path": getattr(logger, "log_path", artifacts.log_artifact_path("test_server.log")),
        },
    },
)


def _tree_profile(tree):
    counts = Counter(node.operator for node in tree)
    aqe_modes = Counter(
        node.data.get("mode")
        for node in tree
        if node.operator == "AQEShuffleRead" and node.data.get("mode")
    )
    scan_nodes = [node for node in tree if node.operator == "Scan"]
    predicate_count = sum(len(node.data.get("predicates", [])) for node in scan_nodes)
    projected_column_count = sum(len(node.data.get("columns", [])) for node in scan_nodes)
    tables = sorted({table for node in tree for table in node.tables})
    root = tree[1] if len(tree) > 1 else None
    return {
        "counts": counts,
        "aqe_modes": aqe_modes,
        "predicate_count": predicate_count,
        "projected_column_count": projected_column_count,
        "tables": tables,
        "table_count": len(tables),
        "root_card": root.card if root is not None else -1,
        "root_size": root.size_in_bytes if root is not None else -1,
    }


def _stack_broadcast_adjustment(tree, request_type):
    profile = _tree_profile(tree)
    counts = profile["counts"]
    aqe_modes = profile["aqe_modes"]
    adjustment = 0.0

    if request_type == 0:
        adjustment += 2.5 * max(profile["root_card"], 0)
        adjustment += 1.0 * max(profile["root_size"], 0)
        adjustment += 3.0 * max(profile["table_count"] - 1, 0)
        adjustment -= 1.25 * profile["predicate_count"]
        adjustment += 4.0 * counts["SortMergeJoin"]
        adjustment += 1.5 * counts["Exchange"]
        adjustment -= 4.0 * counts["BroadcastHashJoin"]
        adjustment -= 1.0 * counts["BroadcastExchange"]
    else:
        adjustment += 6.0 * counts["SortMergeJoin"]
        adjustment += 1.5 * counts["Exchange"]
        adjustment += 0.5 * counts["AQEShuffleRead"]
        adjustment -= 5.0 * counts["BroadcastHashJoin"]
        adjustment -= 1.5 * counts["BroadcastExchange"]
        adjustment -= 0.5 * aqe_modes["local"]
        adjustment -= 0.25 * aqe_modes["coalesce"]

    return adjustment, profile


def _floor_costs(costs):
    return [max(float(cost), MIN_PREDICTION_SECONDS) for cost in costs]


def _shape_costs(trees, raw_costs, request_type):
    if args.cost_shaping_profile == "none":
        return _floor_costs(raw_costs), [0.0] * len(raw_costs), []

    if args.cost_shaping_profile == "stack_broadcast":
        details = [_stack_broadcast_adjustment(tree, request_type) for tree in trees]
        adjustments = [detail[0] for detail in details]
        shaped_costs = [cost + adjustment for cost, adjustment in zip(raw_costs, adjustments)]
        return _floor_costs(shaped_costs), adjustments, [detail[1] for detail in details]

    return _floor_costs(raw_costs), [0.0] * len(raw_costs), []


def _predict_raw_costs(flattened_trees):
    pred = model(flattened_trees)
    return _floor_costs(pred.squeeze(dim=1).tolist())


@app.post("/cost")
async def receive_plan(request: CostRequest):
    trees = [preprocessor.plan2tree(plan_info) for plan_info in request.candidates]
    encoded_tree = [encoder.featurize(tree) for tree in trees]
    flattened_trees = flatten_tree_batch_for_tree_lru(encoded_tree)
    with torch.no_grad():
        raw_costs = _predict_raw_costs(flattened_trees)
        costs, adjustments, profiles = _shape_costs(trees, raw_costs, request.type)
        if request.type == 0:
            logger.info(f"Logical Cost: {costs}")
        elif request.type == 1:
            logger.info(f"Physical Cost: {costs}")
        elif request.type == 2:
            logger.info(f"AQE Cost: {costs}")
        if any(adjustments):
            logger.info(
                "Cost shaping applied: type=%s floored_raw=%s adjustments=%s floored_shaped=%s profiles=%s",
                request.type,
                raw_costs,
                adjustments,
                costs,
                profiles,
            )
    return CostResponse(costs=costs)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=args.port)
