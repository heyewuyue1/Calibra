from fastapi import FastAPI
from request_models import CostRequest, CostResponse, RegisterRequest
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
    tree = preprocessor.plan2tree(request.candidates[0])
    if len(tree) > 3:
        if request.type == 0:  # only save the last full logical plan received for each query
            last_logical_plan_info = (request.candidates[0], time.time_ns())
            return CostResponse(costs=[0])
        
        elif request.type == 1:  # we can save last_logical_plan now
            plan_pool.append(last_logical_plan_info)
            logger.info(f"Saved plan: 0 {last_logical_plan_info[0].plan[:100]} {last_logical_plan_info[1]}") # type: ignore
            plan_pool.append((request.candidates[0], time.time_ns()))
            logger.info(f"Saved plan: 1 {request.candidates[0].plan[:100]} {time.time_ns()}")
            costs = [1.0] * len(request.candidates)
            costs[request.advisoryChoose] = 0.0  # only set advisoryChoose to 0
            return CostResponse(costs=costs)

        elif request.type == 2:
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
    global plan_pool, data_collection, last_logical_plan_info
    logger.info(f"Register training plans for {request.sessionName}")
    current_time = time.time_ns()
    for plan_info, time_stamp in plan_pool:
        if request.executionTime > 0: 
            data_collection.append({"x": {"query_id": request.sessionName, "plan_info": plan_info}, "y": current_time - time_stamp})
        else:  # if anything goes wrong
            data_collection.append({"x": {"query_id": request.sessionName, "tree": plan_info}, "y": 300_000_000_000})
    torch.save(data_collection, ServerConfig.data_path)
    plan_pool = []
    last_logical_plan_info = None
    return {}
 
if __name__ == "__main__":
    uvicorn.run("offline_train_server:app", host="localhost", port=10533)
