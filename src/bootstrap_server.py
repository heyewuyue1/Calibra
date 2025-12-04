# check ServerConfig.data_path
from fastapi import FastAPI
from request_models import RegisterRequest, BetterThanRequest, PhysicalRequest
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

@app.post("/register")
async def register_plan(request: RegisterRequest):
    global collection, cnt
    logger.info(f"Register bootstrap plan pairs for {request.sessionName}, total {cnt} pairs")
    torch.save(collection, ServerConfig.data_path)
    return {}

@app.post("/better_than")
async def better_than(request: BetterThanRequest):
    global collection, hash_collection, cnt
    hashed_tuple = (hash_query_plan(request.thisPlan), hash_query_plan(request.otherPlan), 1 if request.original else 0)
    if hashed_tuple in hash_collection or (hashed_tuple[1], hashed_tuple[0], 1 - hashed_tuple[2]) in hash_collection:
        logger.debug(f"Hashed tuple {hashed_tuple} already exists, skip this pair")
        return {"betterThan": request.original}
    if hashed_tuple[0] == hashed_tuple[1]:
        logger.debug(f"Hashed tuple {hashed_tuple} is the same, skip this pair")
        return {"betterThan": request.original}     
    hash_collection.add(hashed_tuple)
    
    this_tree = preprocessor.plan2tree(request.thisPlan)
    this_tree[1].card = request.thisCard
    this_tree[1].size_in_bytes = request.thisSize
    this_join_order = []
    for node in this_tree:
        if node.operator == 'Scan':
            this_join_order.extend(node.tables)

    other_tree = preprocessor.plan2tree(request.otherPlan)
    other_tree[1].card = request.otherCard
    other_tree[1].size_in_bytes = request.otherSize
    other_join_order = []
    for node in other_tree:
        if node.operator == 'Scan':
            other_join_order.extend(node.tables)
    logger.debug(f"This join order: {this_join_order}, Other join order: {other_join_order}, Native judgement: {request.original}")
    collection.append((this_tree, other_tree, 1 if request.original else 0))
    cnt += 1
    return {"betterThan": request.original}

@app.post('/physical')
async def physical_rank(request: PhysicalRequest):
    global collection, hash_collection, cnt
    first_plan = preprocessor.plan2tree(request.candidates[0])
    if len(first_plan) > 3:
        logger.debug(f"Ranking {len(request.candidates)} plans... ")
        for i in range(1, len(request.candidates)):
            hashed_tuple = (hash_query_plan(request.candidates[0]), hash_query_plan(request.candidates[i]), 1)
            if hashed_tuple in hash_collection:
                logger.debug(f"Hashed tuple {hashed_tuple} already exists, skip this pair")
                continue
            hash_collection.add(hashed_tuple)
            plan = preprocessor.plan2tree(request.candidates[i])
            cnt += 1
            collection.append((first_plan, plan, 1))
    return {"bestIndex": 0}
 
if __name__ == "__main__":
    uvicorn.run("bootstrap_server:app", host="localhost", port=10533)