from pydantic import BaseModel
from typing import List, Dict

class QueryStageInfo(BaseModel):
    materialized: bool  # materialized or not
    card: int           # true/estimated cardinality
    size: int           # true/estimated size in bytes
    stagePlan: str      # physical plan

class PlanInfo(BaseModel):
    plan: str                               
    queryStages: Dict[str, QueryStageInfo]  # key=stage name
    card: int                               # estimated cardinality
    size: int                               # estimated size in bytes

class CostRequest(BaseModel):
    type: int                   # 0-logical plan, 1-physical plan, 2-partially executed plan
    candidates: List[PlanInfo]  # list of candidate
    advisoryChoose: int         # index of Spark SQL's choose

class RegisterRequest(BaseModel):
    sessionName: str
    finalPlan: str      # physical
    executionTime: int  # in nano seconds

class CostResponse(BaseModel):
    costs: List[float]  # list of the costs of the candidates
