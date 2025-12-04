from fastapi import FastAPI
from request_models import CostRequest, RegisterRequest, PhysicalRequest
import uvicorn
from utils.logger import setup_custom_logger
from config import ServerConfig
from preprocessor.simple_preprocessor import SparkPlanPreprocessor
import time
import torch

preprocessor = SparkPlanPreprocessor()

app = FastAPI()
logger = setup_custom_logger("SERVER")

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
            logger.info(f"Saved plan: {request.planType} {request.plan[:100]} {request.queryStages} {time.time_ns()}")
    return {"cost": 0}

@app.post("/register")
async def register_plan(request: RegisterRequest):
    global plan_pool, data_collection, last_logical_plan_info
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
 
if __name__ == "__main__":
    uvicorn.run("train_server:app", host="localhost", port=10533)
