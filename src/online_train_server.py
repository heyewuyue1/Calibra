from fastapi import FastAPI
from request_models import CostRequest, RegisterRequest, PhysicalRequest, BetterThanRequest
import uvicorn
from utils.logger import setup_custom_logger
from utils.util import flatten_tree_batch
from config import ServerConfig
from preprocessor.simple_preprocessor import SparkPlanPreprocessor
import time
import torch
import math
import random
from models.encoder import UnifiedEncoder
from models.TreeLRUNet import TreeLRUNet
from config import TrainConfig

preprocessor = SparkPlanPreprocessor()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = UnifiedEncoder()
model = TreeLRUNet(in_features=encoder.in_features).to(device)
model.load_state_dict(torch.load(TrainConfig.bs_model_save_path))

app = FastAPI()
logger = setup_custom_logger("SERVER")
logger.info(f"Loaded model from {TrainConfig.bs_model_save_path}")

plan_pool = []
data_collection = []
last_logical_plan_info = None

@app.post("/cost")
async def receive_plan(request: CostRequest):
    global last_logical_plan_info, plan_pool
    tree = preprocessor.plan2tree(request.plan, request.queryStages)
    if len(tree) > 3:
        if request.planType == 0:
            last_logical_plan_info = (request.planType, request.plan, request.queryStages, time.time_ns())
        else:
            plan_pool.append((request.planType, request.plan, request.queryStages, time.time_ns()))
            logger.info(f"Saved plan: {request.planType} {request.plan[:100]} {time.time_ns()}")
    return {"cost": 0}

@app.post("/register")
async def register_plan(request: RegisterRequest):
    global plan_pool, data_collection, last_logical_plan_info
    if plan_pool != []:
        logger.info(f"Register training plans for {request.sessionName}")
    current_time = time.time_ns()
    for plan_type, plan, query_stages, time_stamp in plan_pool:
        if request.executionTime > 0: 
            data_collection.append({"x": {"query_id": request.sessionName, "plan_type": plan_type, "plan": plan, "query_stages": query_stages}, "y": current_time - time_stamp})
        else:
            data_collection.append({"x": {"query_id": request.sessionName, "plan_type": plan_type, "plan": plan, "query_stages": query_stages}, "y": 300_000_000_000})
    torch.save(data_collection, ServerConfig.data_path)
    plan_pool = []
    last_logical_plan_info = None
    return {}

@app.post('/physical')
async def physical_rank(request: PhysicalRequest):
    global plan_pool, last_logical_plan_info
    logger.debug(f"Ranking {len(request.candidates)} plans...")
    tree = preprocessor.plan2tree(request.candidates[0])
    if len(tree) > 3:
        plan_pool.append(last_logical_plan_info)
        logger.info(f"Saved plan: {last_logical_plan_info[0]} {last_logical_plan_info[1][:100]} {last_logical_plan_info[3]}")
        plan_pool.append((1, request.candidates[0], {}, time.time_ns()))
        logger.info(f"Saved plan: 1 {request.candidates[0][:100]} {time.time_ns()}")
    return {"bestIndex": 0}

@app.post("/better_than")
async def better_than(request: BetterThanRequest):
    this_tree = preprocessor.plan2tree(request.thisPlan, {})
    this_tree[1].card = request.thisCard
    this_tree[1].size_in_bytes = request.thisSize
    this_join_order = []
    for node in this_tree:
        if node.operator == 'Scan':
            this_join_order.extend(node.tables)

    other_tree = preprocessor.plan2tree(request.otherPlan, {})
    other_tree[1].card = request.otherCard
    other_tree[1].size_in_bytes = request.otherSize
    other_join_order = []
    for node in other_tree:
        if node.operator == 'Scan':
            other_join_order.extend(node.tables)

    this_encoded = encoder.featurize(this_tree, 1)
    other_encoded = encoder.featurize(other_tree, 1)
    flattened_trees = flatten_tree_batch([this_encoded, other_encoded])

    with torch.no_grad():
        pred = model(flattened_trees)
        this_estimated_cost, other_estimated_cost = pred[0].item(), pred[1].item()
        # choose plan base on probability of sigmoid function
        probability = 1 / (1 + math.exp(-this_estimated_cost + other_estimated_cost))
        choose = random.random() < probability
        if choose != request.original:
            if choose:
                logger.debug(f"This: {this_join_order}({this_estimated_cost}), Other: {other_join_order}({other_estimated_cost}).Probability: {probability}, Choose this plan.")
            else:
                logger.debug(f"This: {this_join_order}({this_estimated_cost}), Other: {other_join_order}({other_estimated_cost}).Probability: {probability}, Choose other plan.")
        return {"betterThan": choose}
 
if __name__ == "__main__":
    uvicorn.run("online_train_server:app", host="localhost", port=10533)
