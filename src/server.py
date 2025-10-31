from fastapi import FastAPI
from pydantic import BaseModel
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

# 定义请求体的数据结构
class CostRequest(BaseModel):
    planType: int
    plan: str
    queryStages: dict

class RegisterRequest(BaseModel):
    sessionName: str
    finalPlan: str
    executionTime: int

@app.post("/cost")
async def receive_plan(request: CostRequest):
    tree = preprocessor.plan2tree(request.plan, request.queryStages)
    if ServerConfig.mode == "train" and len(tree) > 3:
        logger.info(f"Evaluating cost for plan: {request.plan}\n{request.queryStages}")
        preprocessor.print_tree(tree, 1)
        plan_pool.append((request.planType, request.plan, request.queryStages, time.time_ns()))
        logger.info(f"Saved plan: {request.planType} {request.plan[:100]} {time.time_ns()}")
    return {"cost": 0}

@app.post("/register")
async def register_plan(request: RegisterRequest):
    global plan_pool, data_collection
    if ServerConfig.mode == "train":
        logger.info(f"Register plans for {request.sessionName}")
        current_time = time.time_ns()
        for plan_type, plan, query_stages, time_stamp in plan_pool:
            data_collection.append({"x": {"query_id": request.sessionName, "plan_type": plan_type, "plan": plan, "query_stages": query_stages}, "y": current_time - time_stamp})
        torch.save(data_collection, ServerConfig.data_path)
        plan_pool = []
    return {}
 
if __name__ == "__main__":
    uvicorn.run("server:app", host="localhost", port=10533)