import argparse
import hashlib
import json
import os

from fastapi import FastAPI
import time
import torch
import uvicorn

from config import DEFAULT_BENCHMARK, ensure_parent_dir, get_run_artifacts, update_manifest
from preprocessor.sparkplanpreprocessor import SparkPlanPreprocessor
from request_models import CostRequest, CostResponse, RegisterRequest
from utils.logger import setup_custom_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--run-id")
    parser.add_argument("--port", type=int, default=10533)
    return parser.parse_args()


args = parse_args()
logger = setup_custom_logger("SERVER")
preprocessor = SparkPlanPreprocessor()
artifacts = get_run_artifacts(args.benchmark, args.run_id)

app = FastAPI()



def get_plan_hash(plan_info):
    if hasattr(plan_info, "model_dump"):
        plan_info_dict = plan_info.model_dump()
    else:
        plan_info_dict = plan_info

    query_stages = plan_info_dict.get("queryStages", {})
    payload = {
        "plan": plan_info_dict.get("plan"),
        "card": plan_info_dict.get("card"),
        "size": plan_info_dict.get("size"),
        "queryStages": {
            key: query_stages[key]
            for key in sorted(query_stages)
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()



def load_existing_data_collection(path):
    if not os.path.exists(path):
        return [], set()

    records = torch.load(path)
    hashes = set()
    for record in records:
        try:
            hashes.add(get_plan_hash(record["x"]["plan_info"]))
        except (KeyError, TypeError):
            continue
    return records, hashes


plan_pool = []
data_collection, data_collection_hashes = load_existing_data_collection(artifacts.raw_training_data_path)
last_logical_plan_info = None


@app.post("/cost")
async def receive_plan(request: CostRequest):
    global last_logical_plan_info, plan_pool
    tree = preprocessor.plan2tree(request.candidates[0])
    if len(tree) > 3:
        if request.type == 0:
            last_logical_plan_info = (request.candidates[0], time.time_ns())
            return CostResponse(costs=[0])

        if request.type == 1:
            plan_pool.append((request.candidates[0], time.time_ns()))
            logger.info(f"Saved plan: 1 {request.candidates[0].plan[:100]} {time.time_ns()}")
            costs = [1.0] * len(request.candidates)
            costs[request.advisoryChoose] = 0.0
            return CostResponse(costs=costs)

        if request.type == 2:
            if last_logical_plan_info is not None:
                plan_pool.append(last_logical_plan_info)
                logger.info(
                    f"Saved plan: 0 {last_logical_plan_info[0].plan[:100]} {last_logical_plan_info[1]}"
                )
                last_logical_plan_info = None
            plan_pool.append((request.candidates[0], time.time_ns()))
            clean_query_stages = {
                k: v.model_dump(exclude={"stagePlan"})
                for k, v in request.candidates[0].queryStages.items()
            }

            logger.info(
                f"Saved plan: 2 {request.candidates[0].plan[:100]} {clean_query_stages} {time.time_ns()}"
            )
            return CostResponse(costs=[0])
    return CostResponse(costs=[0])


@app.post("/register")
async def register_plan(request: RegisterRequest):
    global plan_pool, data_collection, data_collection_hashes, last_logical_plan_info
    current_time = time.time_ns()
    saved_cnt = 0
    duplicate_cnt = 0

    if request.executionTime <= 0:
        logger.warning(
            f"Something went wrong while executing {request.sessionName}, still registering training plans though..."
        )

    for plan_info, time_stamp in plan_pool:
        label = current_time - time_stamp if request.executionTime > 0 else 300_000_000_000

        plan_hash = get_plan_hash(plan_info)
        if plan_hash in data_collection_hashes:
            duplicate_cnt += 1
            continue

        data_collection.append(
            {"x": {"query_id": request.sessionName, "plan_info": plan_info}, "y": label}
        )
        data_collection_hashes.add(plan_hash)
        saved_cnt += 1

    save_path = artifacts.raw_training_data_path
    ensure_parent_dir(save_path)
    torch.save(data_collection, save_path)
    update_manifest(
        artifacts.manifest_path,
        {
            **artifacts.manifest_defaults(),
            "collection": {
                "raw_training_data_path": save_path,
                "registered_queries": len(data_collection),
                "log_path": getattr(logger, "log_path", artifacts.log_artifact_path("offline_train_server.log")),
            },
        },
    )
    logger.info(
        f"Registered {saved_cnt} new training plans for {request.sessionName}; skipped {duplicate_cnt} duplicates; total stored plans: {len(data_collection)}"
    )
    plan_pool = []
    last_logical_plan_info = None
    return {}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=args.port)
