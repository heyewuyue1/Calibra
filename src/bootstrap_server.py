import argparse

from fastapi import FastAPI
import torch
import uvicorn

from config import DEFAULT_BENCHMARK, ensure_parent_dir, get_run_artifacts, update_manifest
from preprocessor.sparkplanpreprocessor import SparkPlanPreprocessor
from request_models import RegisterRequest, CostRequest, CostResponse
from utils.logger import setup_custom_logger
from utils.util import hash_query_plan


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--run-id")
    parser.add_argument("--port", type=int, default=10533)
    return parser.parse_args()


args = parse_args()
preprocessor = SparkPlanPreprocessor()
artifacts = get_run_artifacts(args.benchmark, args.run_id)

app = FastAPI()
logger = setup_custom_logger("SERVER")

collection = []
hash_collection = set()
cnt = 0


@app.post("/cost")
async def receive_plan(request: CostRequest):
    global collection, hash_collection, cnt
    if request.type == 0:
        hashed_tuple = (
            hash_query_plan(request.candidates[0].plan),
            hash_query_plan(request.candidates[1].plan),
            request.advisoryChoose,
        )
        if hashed_tuple in hash_collection or (hashed_tuple[1], hashed_tuple[0], 1 - hashed_tuple[2]) in hash_collection:
            logger.info(f"Hashed tuple {hashed_tuple} already exists, skip this pair")
            return CostResponse(costs=[0, 1] if request.advisoryChoose == 0 else [1, 0])
        if hashed_tuple[0] == hashed_tuple[1]:
            logger.info(f"Hashed tuple {hashed_tuple} is the same, skip this pair")
            return CostResponse(costs=[0, 1] if request.advisoryChoose == 0 else [1, 0])

        hash_collection.add(hashed_tuple)

        this_tree = preprocessor.plan2tree(request.candidates[0])
        this_join_order = []
        for node in this_tree:
            if node.operator == "Scan":
                this_join_order.extend(node.tables)

        other_tree = preprocessor.plan2tree(request.candidates[1])
        other_tree[1].card = request.candidates[1].card
        other_tree[1].size_in_bytes = request.candidates[1].size
        other_join_order = []
        for node in other_tree:
            if node.operator == "Scan":
                other_join_order.extend(node.tables)
        logger.info(
            f"This join order: {this_join_order}, Other join order: {other_join_order}, Native judgement: {request.advisoryChoose}"
        )
        collection.append((this_tree, other_tree, request.advisoryChoose))
        cnt += 1
        return CostResponse(costs=[0, 1] if request.advisoryChoose == 0 else [1, 0])

    if request.type == 1:
        first_plan = preprocessor.plan2tree(request.candidates[0])
        logger.info(request.candidates[0].plan)
        if len(first_plan) > 3:
            logger.info(f"Ranking {len(request.candidates)} plans... ")
            for i in range(1, len(request.candidates)):
                hashed_tuple = (
                    hash_query_plan(request.candidates[0].plan),
                    hash_query_plan(request.candidates[i].plan),
                    1,
                )
                if hashed_tuple in hash_collection:
                    logger.info(f"Hashed tuple {hashed_tuple} already exists, skip this pair")
                    continue
                hash_collection.add(hashed_tuple)
                plan = preprocessor.plan2tree(request.candidates[i])
                cnt += 1
                collection.append((first_plan, plan, 0))
        return CostResponse(costs=[0.0] + [1.0] * (len(request.candidates) - 1))

    return CostResponse(costs=[0])


@app.post("/register")
async def register_plan(request: RegisterRequest):
    global collection, cnt
    logger.info(f"Register bootstrap plan pairs for {request.sessionName}, total {cnt} pairs")
    save_path = artifacts.bootstrap_data_path
    ensure_parent_dir(save_path)
    torch.save(collection, save_path)
    update_manifest(
        artifacts.manifest_path,
        {
            **artifacts.manifest_defaults(),
            "bootstrap_collection": {
                "pair_count": len(collection),
            },
        },
    )
    return {}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=args.port)
