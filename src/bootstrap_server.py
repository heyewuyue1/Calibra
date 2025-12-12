# check ServerConfig.data_path
from fastapi import FastAPI
from request_models import RegisterRequest, CostRequest, CostResponse
import uvicorn
from utils.logger import setup_custom_logger
from config import ServerConfig
from preprocessor.simple_preprocessor import SparkPlanPreprocessor
import torch
from utils.util import hash_query_plan

preprocessor = SparkPlanPreprocessor()

app = FastAPI()
logger = setup_custom_logger("SERVER")

collection = []  # [(X1, X2, Y)^n]
hash_collection = set() # {(hash(X1), hash(X2), Y)^n}
cnt = 0

@app.post("/cost")
async def receive_plan(request: CostRequest):
    global collection, hash_collection, cnt
    if request.type == 0: # sub-logical plan
        # check if exists first
        hashed_tuple = (hash_query_plan(request.candidates[0].plan), 
                        hash_query_plan(request.candidates[1].plan),
                        request.advisoryChoose)  
                        # advisoryChoose is 0 -> first plan is better -> pred1 - pred2 < 0 -> sigmoid < 0.5 -> labeled as 0
        if hashed_tuple in hash_collection or (hashed_tuple[1], hashed_tuple[0], 1 - hashed_tuple[2]) in hash_collection:
            logger.info(f"Hashed tuple {hashed_tuple} already exists, skip this pair")
            return CostResponse(costs=[0, 1] if request.advisoryChoose == 0 else [1, 0])
        if hashed_tuple[0] == hashed_tuple[1]:
            logger.info(f"Hashed tuple {hashed_tuple} is the same, skip this pair")
            return CostResponse(costs=[0, 1] if request.advisoryChoose == 0 else [1, 0])
        
        # if not, add to hash_collection
        hash_collection.add(hashed_tuple)

        # add to collection
        this_tree = preprocessor.plan2tree(request.candidates[0])
        this_join_order = []
        for node in this_tree:
            if node.operator == 'Scan':
                this_join_order.extend(node.tables)

        other_tree = preprocessor.plan2tree(request.candidates[1])
        other_tree[1].card = request.candidates[1].card
        other_tree[1].size_in_bytes = request.candidates[1].size
        other_join_order = []
        for node in other_tree:
            if node.operator == 'Scan':
                other_join_order.extend(node.tables)
        logger.info(f"This join order: {this_join_order}, Other join order: {other_join_order}, Native judgement: {request.advisoryChoose}")
        collection.append((this_tree, other_tree, request.advisoryChoose))
        cnt += 1
        return CostResponse(costs=[0, 1] if request.advisoryChoose == 0 else [1, 0])

    if request.type == 1: # physical plan
        first_plan = preprocessor.plan2tree(request.candidates[0])
        if len(first_plan) > 3:
            logger.info(f"Ranking {len(request.candidates)} plans... ")
            for i in range(1, len(request.candidates)):
                hashed_tuple = (hash_query_plan(request.candidates[0].plan), hash_query_plan(request.candidates[i].plan), 1)
                if hashed_tuple in hash_collection:
                    logger.info(f"Hashed tuple {hashed_tuple} already exists, skip this pair")
                    continue
                hash_collection.add(hashed_tuple)
                plan = preprocessor.plan2tree(request.candidates[i])
                cnt += 1
                collection.append((first_plan, plan, 0))  # always choose the first plan
        return CostResponse(
            costs=[0.0] + [1.0] * (len(request.candidates) - 1)
        )
        
@app.post("/register")
async def register_plan(request: RegisterRequest):
    global collection, cnt
    logger.info(f"Register bootstrap plan pairs for {request.sessionName}, total {cnt} pairs")
    torch.save(collection, ServerConfig.data_path)
    return {}
 
if __name__ == "__main__":
    uvicorn.run("bootstrap_server:app", host="localhost", port=10533)