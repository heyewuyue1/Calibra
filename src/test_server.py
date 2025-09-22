from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import json
from utils.logger import setup_custom_logger
from config import ServerConfig
from preprocessor.simple_preprocessor import SparkPlanPreprocessor

preprocessor = SparkPlanPreprocessor()

app = FastAPI()
logger = setup_custom_logger("TEST_SERVER")

# 定义请求体的数据结构
class CostRequest(BaseModel):
    plan: str

@app.post("/cost")
async def receive_plan(request: CostRequest):
    plan = request.plan  # 从请求体中提取 plan
    logger.info(f"Evaluating cost for plan: {plan[:100]}")
    plan_json = json.loads(plan)
    root = preprocessor.build_tree(plan_json)
    # preprocessor.print_tree_ascii(root)
    return {"cost": 0}

if __name__ == "__main__":
    uvicorn.run("test_server:app", host="localhost", port=10533)