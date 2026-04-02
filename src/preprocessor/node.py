from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Node:
    operator: str
    executed: int = 0
    tables: List[str] = field(default_factory=list)
    card: float = -1
    size_in_bytes: float = -1
    data: Dict = field(default_factory=dict)
    lc: Optional[int] = None
    rc: Optional[int] = None
