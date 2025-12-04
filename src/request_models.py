from pydantic import BaseModel

# 定义请求体的数据结构
class CostRequest(BaseModel):
    planType: int
    plan: str
    queryStages: dict

class RegisterRequest(BaseModel):
    sessionName: str
    finalPlan: str
    executionTime: int

class BetterThanRequest(BaseModel):
    thisPlan: str
    thisCard: int
    thisSize: int
    otherPlan: str
    otherCard: int
    otherSize: int
    original: bool

class PhysicalRequest(BaseModel):
    candidates: list