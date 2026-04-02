from typing import Any, Dict

from preprocessor.sparkplanpreprocessor import SparkPlanPreprocessor as _SparkPlanPreprocessor
from request_models import PlanInfo, QueryStageInfo


class SparkPlanPreprocessor(_SparkPlanPreprocessor):
    def _normalize_stage_info(self, stage: Any) -> QueryStageInfo:
        if isinstance(stage, QueryStageInfo):
            return stage
        if isinstance(stage, dict):
            return QueryStageInfo(**stage)
        if isinstance(stage, (list, tuple)) and len(stage) >= 4:
            return QueryStageInfo(
                materialized=bool(stage[0]),
                card=int(eval(stage[1])) if isinstance(stage[1], str) else int(stage[1]),
                size=int(eval(stage[2])) if isinstance(stage[2], str) else int(stage[2]),
                stagePlan=stage[3],
            )
        raise TypeError(f"Unsupported query stage info: {stage!r}")

    def plan2tree(self, plan_info, queryStages: Dict[str, Any] = None):
        if isinstance(plan_info, PlanInfo):
            normalized = plan_info
        else:
            raw_query_stages = queryStages or {}
            normalized_query_stages = {
                name: self._normalize_stage_info(stage)
                for name, stage in raw_query_stages.items()
            }
            normalized = PlanInfo(
                plan=plan_info,
                queryStages=normalized_query_stages,
                card=-1,
                size=-1,
            )
        return super().plan2tree(normalized)

    def query_stage2tree(self, tree, i, executed, stage_plan):
        self._parse_stage_plan_into_tree(tree, executed, stage_plan)
        self._fill_join_tables_bottom_up(tree, i - 1)
        return tree


__all__ = ["SparkPlanPreprocessor"]
