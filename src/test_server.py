from fastapi import FastAPI
from request_models import CostRequest, CostResponse
from utils.util import flatten_tree_batch_for_tree_lru
import uvicorn
from utils.logger import setup_custom_logger
from config import TrainConfig
from preprocessor.simple_preprocessor import SparkPlanPreprocessor
from models.encoder import UnifiedFeatureEncoder
from models.TreeLRUNet import TreeLRUNet
import torch

preprocessor = SparkPlanPreprocessor()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = UnifiedFeatureEncoder()
model = TreeLRUNet(in_features=encoder.in_features).to(device)
model.load_state_dict(torch.load(TrainConfig.model_save_path))

app = FastAPI()
logger = setup_custom_logger("SERVER")
logger.info(f"Loaded model from {TrainConfig.model_save_path}")

plan_pool = []
data_collection = []

@app.post("/cost")
async def receive_plan(request: CostRequest):
    trees = [preprocessor.plan2tree(plan_info) for plan_info in request.candidates]
    encoded_tree = [encoder.featurize(tree) for tree in trees]
    flattened_trees = flatten_tree_batch_for_tree_lru(encoded_tree)
    with torch.no_grad():
        pred = model(flattened_trees)
        costs = pred.squeeze(dim=1).tolist()
        if request.type == 0:
            logger.info(f"Logical Cost: {costs}")
        elif request.type == 1:
            logger.info(f"Physical Cost: {costs}")
        elif request.type == 2:
            logger.info(f"AQE Cost: {costs}")
    return CostResponse(costs=costs)
 
if __name__ == "__main__":
    uvicorn.run("test_server:app", host="localhost", port=10533)
