from fastapi import FastAPI
from request_models import CostRequest, BetterThanRequest, PhysicalRequest
from utils.util import flatten_tree_batch
import uvicorn
from utils.logger import setup_custom_logger
from config import TrainConfig
from preprocessor.simple_preprocessor import SparkPlanPreprocessor
from models.encoder import UnifiedEncoder
from models.TreeLRUNet import TreeLRUNet
import torch

preprocessor = SparkPlanPreprocessor()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = UnifiedEncoder()
model = TreeLRUNet(in_features=encoder.in_features).to(device)
model.load_state_dict(torch.load(TrainConfig.model_save_path))

app = FastAPI()
logger = setup_custom_logger("SERVER")
logger.info(f"Loaded model from {TrainConfig.model_save_path}")

plan_pool = []
data_collection = []

@app.post("/cost")
async def receive_plan(request: CostRequest):
    tree = preprocessor.plan2tree(request.plan, request.queryStages)
    encoded_tree = encoder.featurize(tree)
    flattened_trees = flatten_tree_batch([encoded_tree])
    with torch.no_grad():
        pred = model(flattened_trees)
        cost = pred[0].item()
        if request.planType == 0:
            logger.debug(f"CBO Cost: {request.plan[:100]}({cost})")
        elif request.planType == 1:
            logger.debug(f"AQE Cost: {request.plan[:100]}({cost})")
    return {"cost": cost}

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
        calibra_judgement = this_estimated_cost < other_estimated_cost
        if request.original != calibra_judgement:
            logger.debug(f"This: {this_join_order}({this_estimated_cost})")
            logger.debug(f"Other: {other_join_order}({other_estimated_cost})")
        return {"betterThan": this_estimated_cost < other_estimated_cost}

@app.post('/physical')
async def physical_rank(request: PhysicalRequest):
    trees = [preprocessor.plan2tree(raw_plan) for raw_plan in request.candidates]
    encoded_trees = [encoder.featurize(tree) for tree in trees]
    flattened_trees = flatten_tree_batch(encoded_trees)
    with torch.no_grad():
        pred = model(flattened_trees)
        best_index = torch.argmin(pred.squeeze()).item()
    for i, cost in enumerate(pred.tolist()):
        logger.debug(f"Physical Cost: {request.candidates[i]} ({cost})")
    return {"bestIndex": best_index}
 
if __name__ == "__main__":
    uvicorn.run("test_server:app", host="localhost", port=10533)
