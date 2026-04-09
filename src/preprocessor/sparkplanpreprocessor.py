import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from config import EnvironmentConfig
from preprocessor.node import Node
from request_models import PlanInfo
from utils.logger import setup_custom_logger

logger = setup_custom_logger("PREPROCESSOR")

TABLE_RE = re.compile(r"spark_catalog\.[\w\.]+\.([\w_]+)\[")
BRACKET_RE = re.compile(r"\[([^\]]+)\]")
HASH_PARTITION_RE = re.compile(r"hashpartitioning\s*\(.*?,\s*(\d+)\s*\)")
BHJ_RE = re.compile(
    r"\b\w+Join\s*\[([^\]]+)\],\s*\[([^\]]+)\].*?\b(BuildRight|BuildLeft)\b"
)
SMJ_RE = re.compile(r"SortMergeJoin(?:\(skew=true\))?\s*\[([^\]]+)\],\s*\[([^\]]+)\]")
BNLJ_RE = re.compile(r"BroadcastNestedLoopJoin\s+(BuildRight|BuildLeft)")
ATTR_RE = re.compile(r"(\w+)#\d+")
SCAN_COLUMN_RE = re.compile(r"(\w+)#?\d*")
PROJECT_RE = re.compile(r"Project \[([^\]]*)\]")
PLAN_MARKER_RE = re.compile(r"(\+-|:-)\s")
WHOLE_STAGE_RE = re.compile(r"^\*\(\d+\)\s+")

COMPARISON_OPERATORS = ["<=>", ">=", "<=", "!=", "<>", "=", ">", "<"]
COMPARISON_OPERATOR_MAP = {
    "<>": "!=",
    "<=": "<",
    "<": "<",
    ">=": ">",
    ">": ">",
}
FUNCTION_OPERATOR_MAP = {
    "contains": "contains",
    "stringcontains": "contains",
    "startswith": "contains",
    "endswith": "contains",
    "like": "contains",
    "in": "in",
    "inset": "in",
    "equalto": "=",
    "equalnullsafe": "<=>",
    "greaterthan": ">",
    "greaterthanorequal": ">",
    "lessthan": "<",
    "lessthanorequal": "<",
    "notequalto": "!=",
    "isnotnull": "isnotnull",
    "isnull": "isnull",
}
PredicateTriple = List[Optional[str]]

SCAN_CONTEXT_OPS = {
    "AQEShuffleRead",
    "BroadcastExchange",
    "ColumnarToRow",
    "Exchange",
    "Filter",
    "InputAdapter",
    "Project",
    "ReusedExchange",
    "RowToColumnar",
    "WholeStageCodegen",
}


@dataclass
class NodeSpec:
    operator: str
    executed: int = 0
    tables: List[str] = field(default_factory=list)
    card: float = -1
    size_in_bytes: float = -1
    data: Dict = field(default_factory=dict)
    stage_name: Optional[str] = None
    stage_plan: Optional[str] = None


@dataclass
class RawPlanLine:
    line_no: int
    text: str
    depth: int
    operator: Optional[str]
    parent_line_no: Optional[int] = None


@dataclass
class ScanContext:
    predicates: List[PredicateTriple] = field(default_factory=list)
    project_columns: List[str] = field(default_factory=list)


class SparkPlanPreprocessor:
    def __init__(self) -> None:
        self.op_list = [
            "Aggregate",
            "Scan",
            "Join Inner",
            "LogicalQueryStage",
            "SortMergeJoin",
            "BroadcastHashJoin",
            "AQEShuffleRead",
            "Exchange",
            "BroadcastExchange",
        ]
        self.op_dim = len(self.op_list)
        logger.info(f"Considering {self.op_dim} opeartors.")

    def plan2tree(self, plan_info: PlanInfo):
        tree = [Node("Root", 0, [], -1, -1, {})]
        self._parse_plan_into_tree(tree, plan_info)
        self._fill_join_tables_bottom_up(tree, 0)
        self._fill_root_stats(tree, plan_info)
        return tree

    def _parse_plan_into_tree(self, tree: List[Node], plan_info: PlanInfo) -> None:
        lines = plan_info.plan.splitlines()
        scan_contexts = self._build_scan_contexts(lines)
        self._parse_lines_into_tree(
            tree=tree,
            lines=lines,
            parse_line=lambda raw_line, line_no: self._parse_main_line(
                raw_line, line_no, plan_info, scan_contexts
            ),
            root_parent_idx=0,
            error_prefix="",
        )

    def _parse_stage_plan_into_tree(self, tree: List[Node], executed: int, stage_plan: str) -> None:
        lines = stage_plan.splitlines()
        scan_contexts = self._build_scan_contexts(lines)
        self._parse_lines_into_tree(
            tree=tree,
            lines=lines,
            parse_line=lambda raw_line, line_no: self._parse_stage_line(
                raw_line, line_no, executed, scan_contexts
            ),
            root_parent_idx=len(tree) - 1,
            error_prefix="stage ",
        )

    def _parse_lines_into_tree(
        self,
        tree: List[Node],
        lines: List[str],
        parse_line,
        root_parent_idx: int,
        error_prefix: str,
    ) -> None:
        raw_lines = self._build_raw_plan_lines(lines)
        raw_by_line_no = {raw_line.line_no: raw_line for raw_line in raw_lines}
        node_idx_by_line_no: Dict[int, int] = {}

        for raw_line in raw_lines:
            spec = parse_line(raw_line.text, raw_line.line_no)
            if spec is None:
                continue

            tree.append(self._build_node(spec))
            cur_idx = len(tree) - 1
            node_idx_by_line_no[raw_line.line_no] = cur_idx

            parent_idx = self._find_parent_node_idx(raw_line, raw_by_line_no, node_idx_by_line_no)
            if parent_idx is None:
                parent_idx = root_parent_idx

            parent = tree[parent_idx]
            if parent.lc is None:
                parent.lc = cur_idx
            elif parent.rc is None:
                parent.rc = cur_idx
            else:
                raise ValueError(
                    f"Unexpected extra child for {error_prefix}parent line: {raw_by_line_no[raw_line.parent_line_no].text if raw_line.parent_line_no is not None else 'ROOT'}"
                )

            if spec.stage_plan:
                self._parse_stage_plan_into_tree(tree, spec.executed, spec.stage_plan)

    def _find_parent_node_idx(
        self,
        raw_line: RawPlanLine,
        raw_by_line_no: Dict[int, RawPlanLine],
        node_idx_by_line_no: Dict[int, int],
    ) -> Optional[int]:
        parent_line_no = raw_line.parent_line_no
        while parent_line_no is not None:
            parent_idx = node_idx_by_line_no.get(parent_line_no)
            if parent_idx is not None:
                return parent_idx
            parent_line_no = raw_by_line_no[parent_line_no].parent_line_no
        return None

    def _parse_main_line(
        self,
        line: str,
        line_no: int,
        plan_info: PlanInfo,
        scan_contexts: Dict[int, ScanContext],
    ) -> Optional[NodeSpec]:
        return (
            self._parse_aggregate(line)
            or self._parse_relation(line, line_no, scan_contexts)
            or self._parse_scan(line, line_no, scan_contexts)
            or self._parse_join_inner(line)
            or self._parse_sort_merge_join(line)
            or self._parse_broadcast_hash_join(line)
            or self._parse_broadcast_nested_loop_join(line)
            or self._parse_shuffled_hash_join(line)
            or self._parse_cartesian_product(line)
            or self._parse_logical_query_stage(line, plan_info)
        )

    def _parse_stage_line(
        self,
        line: str,
        line_no: int,
        executed: int,
        scan_contexts: Dict[int, ScanContext],
    ) -> Optional[NodeSpec]:
        return (
            self._parse_broadcast_exchange(line, executed=executed)
            or self._parse_exchange(line, executed=executed)
            or self._parse_sort_merge_join(line, executed=executed)
            or self._parse_broadcast_hash_join(line, executed=executed)
            or self._parse_broadcast_nested_loop_join(line, executed=executed)
            or self._parse_shuffled_hash_join(line, executed=executed)
            or self._parse_cartesian_product(line, executed=executed)
            or self._parse_aqe_shuffle_read(line, executed=executed)
            or self._parse_scan(line, line_no, scan_contexts, executed=executed)
        )

    def _build_node(self, spec: NodeSpec) -> Node:
        return Node(
            spec.operator,
            spec.executed,
            spec.tables,
            spec.card,
            spec.size_in_bytes,
            spec.data,
        )

    def _parse_aggregate(self, line: str) -> Optional[NodeSpec]:
        if "Aggregate" not in line:
            return None
        return NodeSpec(operator="Aggregate")

    def _parse_relation(
        self,
        line: str,
        line_no: int,
        scan_contexts: Dict[int, ScanContext],
    ) -> Optional[NodeSpec]:
        if "Relation" not in line:
            return None

        if "HiveTableRelation" in line:
            tables = [line.split("`")[-2]]
        else:
            tables = [line.split("[")[0].split(".")[-1]]

        context = scan_contexts.get(line_no, ScanContext())
        columns = context.project_columns or self._extract_relation_columns(line)
        predicates = self._qualify_predicates(tables[0] if tables else None, context.predicates)

        return NodeSpec(
            operator="Scan",
            tables=tables,
            data={"columns": columns, "predicates": predicates},
        )

    def _parse_scan(
        self,
        line: str,
        line_no: int,
        scan_contexts: Dict[int, ScanContext],
        executed: int = 0,
    ) -> Optional[NodeSpec]:
        if "FileScan" not in line:
            return None

        table_match = TABLE_RE.search(line)
        table = table_match.group(1) if table_match else None
        tables = [table] if table else []

        cols_match = BRACKET_RE.search(line)
        columns = SCAN_COLUMN_RE.findall(cols_match.group(1)) if cols_match else []
        columns = list(dict.fromkeys(columns))

        context = scan_contexts.get(line_no, ScanContext())
        predicates = context.predicates or self._extract_data_filters(line)
        predicates = self._qualify_predicates(table, predicates)

        return NodeSpec(
            operator="Scan",
            executed=executed,
            tables=tables,
            data={"columns": columns, "predicates": predicates},
        )

    def _parse_join_inner(self, line: str) -> Optional[NodeSpec]:
        if "Join Inner" not in line:
            return None
        return NodeSpec(operator="Join Inner")

    def _parse_sort_merge_join(self, line: str, executed: int = 0) -> Optional[NodeSpec]:
        if "SortMergeJoin" not in line:
            return None
        match = SMJ_RE.search(line)
        data: Dict[str, Optional[str]] = {}
        if match:
            left_keys = self._extract_attr_names(match.group(1))
            right_keys = self._extract_attr_names(match.group(2))
            data = {
                "left": left_keys[0] if left_keys else None,
                "right": right_keys[0] if right_keys else None,
            }
        return NodeSpec(operator="SortMergeJoin", executed=executed, data=data)

    def _parse_broadcast_hash_join(self, line: str, executed: int = 0) -> Optional[NodeSpec]:
        if "BroadcastHashJoin" not in line:
            return None
        match = BHJ_RE.search(line)
        data: Dict[str, Optional[str]] = {}
        if match:
            left_keys = self._extract_attr_names(match.group(1))
            right_keys = self._extract_attr_names(match.group(2))
            data = {
                "left": left_keys[0] if left_keys else None,
                "right": right_keys[0] if right_keys else None,
                "build_side": match.group(3),
            }
        return NodeSpec(operator="BroadcastHashJoin", executed=executed, data=data)

    def _parse_broadcast_nested_loop_join(self, line: str, executed: int = 0) -> Optional[NodeSpec]:
        if "BroadcastNestedLoopJoin" not in line:
            return None
        match = BNLJ_RE.search(line)
        data: Dict[str, Optional[str]] = {}
        if match:
            data = {"build_side": match.group(1)}
        return NodeSpec(operator="BroadcastNestedLoopJoin", executed=executed, data=data)

    def _parse_shuffled_hash_join(self, line: str, executed: int = 0) -> Optional[NodeSpec]:
        if "ShuffledHashJoin" not in line:
            return None
        match = BHJ_RE.search(line)
        data: Dict[str, Optional[str]] = {}
        if match:
            left_keys = self._extract_attr_names(match.group(1))
            right_keys = self._extract_attr_names(match.group(2))
            data = {
                "left": left_keys[0] if left_keys else None,
                "right": right_keys[0] if right_keys else None,
                "build_side": match.group(3),
            }
        return NodeSpec(operator="ShuffledHashJoin", executed=executed, data=data)

    def _parse_cartesian_product(self, line: str, executed: int = 0) -> Optional[NodeSpec]:
        if "CartesianProduct" not in line:
            return None
        return NodeSpec(operator="CartesianProduct", executed=executed)

    def _safe_log_stat(self, value: float) -> float:
        if value is None or value < 0:
            return -1
        logged = np.log1p(value)
        if np.isfinite(logged):
            return logged
        return -1

    def _parse_logical_query_stage(self, line: str, plan_info: PlanInfo) -> Optional[NodeSpec]:
        if "LogicalQueryStage" not in line:
            return None

        stage_name = line.split("- ")[-1].strip()
        stage_info = plan_info.queryStages.get(stage_name)
        if stage_info is None:
            logger.warning(f"Missing query stage details for {stage_name}")
            return NodeSpec(operator="LogicalQueryStage", stage_name=stage_name)

        return NodeSpec(
            operator="LogicalQueryStage",
            executed=1 if stage_info.materialized else 0,
            card=self._safe_log_stat(stage_info.card),
            size_in_bytes=self._safe_log_stat(stage_info.size),
            stage_name=stage_name,
            stage_plan=stage_info.stagePlan,
        )

    def _parse_broadcast_exchange(self, line: str, executed: int = 0) -> Optional[NodeSpec]:
        if "BroadcastExchange" not in line:
            return None
        return NodeSpec(operator="BroadcastExchange", executed=executed)

    def _parse_exchange(self, line: str, executed: int = 0) -> Optional[NodeSpec]:
        if "Exchange" not in line or "BroadcastExchange" in line:
            return None
        partition_match = HASH_PARTITION_RE.search(line)
        data = {
            "key": self._extract_exchange_key(line),
            "partition_number": int(partition_match.group(1)) if partition_match else None,
        }
        return NodeSpec(operator="Exchange", executed=executed, data=data)

    def _parse_aqe_shuffle_read(self, line: str, executed: int = 0) -> Optional[NodeSpec]:
        if "AQEShuffleRead" not in line:
            return None
        return NodeSpec(
            operator="AQEShuffleRead",
            executed=executed,
            data={"mode": line.split("AQEShuffleRead ")[-1]},
        )

    def _get_plan_depth(self, line: str) -> int:
        return line.split("+-")[0].count(":")

    def _get_stage_depth(self, line: str) -> int:
        return line.count(":")

    def _extract_attr_names(self, text: str) -> List[str]:
        return list(dict.fromkeys(ATTR_RE.findall(text)))

    def _extract_exchange_key(self, line: str) -> Optional[str]:
        try:
            return line.split("(")[-1].split(")")[0].split(", ")[0].split("#")[0]
        except Exception:
            return None

    def _build_scan_contexts(self, lines: List[str]) -> Dict[int, ScanContext]:
        raw_lines = self._build_raw_plan_lines(lines)
        raw_by_line_no = {raw_line.line_no: raw_line for raw_line in raw_lines}
        contexts: Dict[int, ScanContext] = {}

        for raw_line in raw_lines:
            if raw_line.operator not in {"FileScan", "Relation"}:
                continue

            predicates: List[PredicateTriple] = []
            project_columns: List[str] = []
            parent_line_no = raw_line.parent_line_no

            while parent_line_no is not None:
                parent = raw_by_line_no[parent_line_no]
                if parent.operator == "Project" and not project_columns:
                    project_columns = self._extract_project_columns(parent.text)
                if parent.operator == "Filter":
                    predicates.extend(self._extract_filter_predicates(parent.text))
                if parent.operator not in SCAN_CONTEXT_OPS:
                    break
                parent_line_no = parent.parent_line_no

            contexts[raw_line.line_no] = ScanContext(
                predicates=self._dedupe_predicate_triples(predicates),
                project_columns=project_columns,
            )

        return contexts

    def _build_raw_plan_lines(self, lines: List[str]) -> List[RawPlanLine]:
        raw_lines: List[RawPlanLine] = []
        stack: List[RawPlanLine] = []

        for line_no, line in enumerate(lines):
            if not line.strip():
                continue

            depth = self._get_raw_depth(line)
            while stack and stack[-1].depth >= depth:
                stack.pop()

            raw_line = RawPlanLine(
                line_no=line_no,
                text=line,
                depth=depth,
                operator=self._extract_raw_operator(line),
                parent_line_no=stack[-1].line_no if stack else None,
            )
            raw_lines.append(raw_line)
            stack.append(raw_line)

        return raw_lines

    def _get_raw_depth(self, line: str) -> int:
        marker = PLAN_MARKER_RE.search(line)
        if marker is None:
            return 0
        return marker.start() // 3 + 1

    def _extract_raw_operator(self, line: str) -> Optional[str]:
        stripped = self._strip_plan_prefix(line)
        match = re.match(r"([A-Za-z][A-Za-z0-9]*)", stripped)
        return match.group(1) if match else None

    def _strip_plan_prefix(self, line: str) -> str:
        marker = PLAN_MARKER_RE.search(line)
        stripped = line[marker.end() :] if marker else line
        stripped = stripped.strip()
        return WHOLE_STAGE_RE.sub("", stripped)

    def _extract_project_columns(self, line: str) -> List[str]:
        stripped = self._strip_plan_prefix(line)
        match = PROJECT_RE.search(stripped)
        if match is None:
            return []

        columns = [
            self._normalize_identifier(column.split(" AS ", 1)[0])
            for column in self._split_top_level_items(match.group(1), ",")
        ]
        return self._dedupe_keep_order(columns)

    def _extract_relation_columns(self, line: str) -> List[str]:
        match = BRACKET_RE.search(line)
        if match is None:
            return []
        return self._dedupe_keep_order(SCAN_COLUMN_RE.findall(match.group(1)))

    def _extract_filter_predicates(self, line: str) -> List[PredicateTriple]:
        stripped = self._strip_plan_prefix(line)
        if "Filter " not in stripped:
            return []
        expression = stripped.split("Filter ", 1)[1].strip()
        return self._structure_predicates(self._split_conjuncts(expression))

    def _split_conjuncts(self, expression: str) -> List[str]:
        expression = self._strip_outer_parentheses(expression.strip())
        if not expression:
            return []

        for delimiter in ("AND", "OR"):
            parts = self._split_top_level_items(expression, delimiter)
            if len(parts) > 1:
                predicates: List[str] = []
                for part in parts:
                    predicates.extend(self._split_conjuncts(part))
                return self._dedupe_keep_order(predicates)

        return [self._normalize_predicate(expression)]

    def _structure_predicates(self, predicates: List[str]) -> List[PredicateTriple]:
        structured: List[PredicateTriple] = []
        for predicate in predicates:
            structured.extend(self._predicate_to_triplets(predicate))
        return self._dedupe_predicate_triples(structured)

    def _predicate_to_triplets(self, predicate: str) -> List[PredicateTriple]:
        predicate = self._normalize_predicate(predicate)
        if not predicate:
            return []

        if predicate.upper().startswith("NOT "):
            negated = predicate[4:].strip()
            function_call = self._parse_function_call(negated)
            if function_call is not None and function_call[0].lower() in {"contains", "stringcontains"}:
                predicate = negated

        keyword_predicate = self._split_keyword_predicate(predicate)
        if keyword_predicate is not None:
            left, operator, right = keyword_predicate
            column = self._extract_column_name(left)
            operator = self._normalize_operator(operator)
            if operator == "isnull":
                return []
            expanded = self._expand_set_predicate(column, operator, right)
            if expanded:
                return expanded
            return [[column, operator, self._normalize_constant(right)]]

        comparison = self._split_comparison_predicate(predicate)
        if comparison is not None:
            left, operator, right = comparison
            return [[
                self._extract_column_name(left),
                self._normalize_operator(operator),
                self._normalize_constant(right),
            ]]

        function_call = self._parse_function_call(predicate)
        if function_call is not None:
            func_name, args = function_call
            if not args:
                return []

            column = self._extract_column_name(args[0])
            operator = self._normalize_operator(
                FUNCTION_OPERATOR_MAP.get(func_name.lower(), func_name.lower())
            )
            if operator == "isnull":
                return []
            if operator == "in":
                expanded = self._expand_set_values(column, args[1:])
                if expanded:
                    return expanded

            value = None
            if len(args) > 1:
                normalized_values = [
                    normalized
                    for normalized in (self._normalize_constant(arg) for arg in args[1:])
                    if normalized
                ]
                value = ", ".join(normalized_values) if normalized_values else None

            return [[column, operator, value]]

        cleaned = self._remove_all_brackets(predicate)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return [[cleaned, "raw", None]] if cleaned else []

    def _predicate_to_triplet(self, predicate: str) -> Optional[PredicateTriple]:
        triplets = self._predicate_to_triplets(predicate)
        return triplets[0] if triplets else None

    def _expand_set_predicate(
        self,
        column: str,
        operator: str,
        raw_values: str,
    ) -> List[PredicateTriple]:
        if operator != "in":
            return []
        values = self._split_set_values(raw_values)
        return [[column, "=", value] for value in values]

    def _expand_set_values(
        self,
        column: str,
        raw_values: List[str],
    ) -> List[PredicateTriple]:
        values: List[str] = []
        for raw_value in raw_values:
            values.extend(self._split_set_values(raw_value))
        return [[column, "=", value] for value in values]

    def _split_set_values(self, raw_values: str) -> List[str]:
        normalized_values = self._normalize_predicate(raw_values)
        normalized_values = self._strip_outer_parentheses(normalized_values)
        parts = self._split_top_level_items(normalized_values, ",")
        values = [
            normalized
            for normalized in (self._normalize_constant(part) for part in parts)
            if normalized
        ]
        return self._dedupe_keep_order(values)

    def _split_keyword_predicate(self, predicate: str):
        paren_depth = 0
        bracket_depth = 0
        idx = 0

        while idx < len(predicate):
            if paren_depth == 0 and bracket_depth == 0:
                for keyword in ("INSET", "LIKE", "IN"):
                    token = f" {keyword} "
                    if predicate.startswith(token, idx):
                        left = predicate[:idx].strip()
                        right = predicate[idx + len(token) :].strip()
                        if left and right:
                            return left, self._normalize_operator(keyword.lower()), right

            char = predicate[idx]
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth -= 1
            elif char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth -= 1
            idx += 1

        return None

    def _split_comparison_predicate(self, predicate: str):
        paren_depth = 0
        bracket_depth = 0
        idx = 0

        while idx < len(predicate):
            if paren_depth == 0 and bracket_depth == 0:
                for operator in COMPARISON_OPERATORS:
                    if predicate.startswith(operator, idx):
                        left = predicate[:idx].strip()
                        right = predicate[idx + len(operator) :].strip()
                        if left and right:
                            normalized = self._normalize_operator(
                                COMPARISON_OPERATOR_MAP.get(operator, operator)
                            )
                            return left, normalized, right

            char = predicate[idx]
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth -= 1
            elif char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth -= 1
            idx += 1

        return None

    def _parse_function_call(self, expression: str):
        expression = self._strip_outer_parentheses(expression.strip())
        open_idx = expression.find("(")
        if open_idx <= 0 or not expression.endswith(")"):
            return None

        func_name = expression[:open_idx].strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", func_name):
            return None

        depth = 0
        for idx in range(open_idx, len(expression)):
            char = expression[idx]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            if depth == 0 and idx != len(expression) - 1:
                return None

        if depth != 0:
            return None

        args_text = expression[open_idx + 1 : -1].strip()
        args = self._split_top_level_items(args_text, ",") if args_text else []
        return func_name, args

    def _extract_column_name(self, expression: str) -> str:
        expression = self._normalize_predicate(expression)
        if not expression:
            return ""

        function_call = self._parse_function_call(expression)
        if function_call is not None:
            func_name, args = function_call
            if args:
                if func_name.lower() == "cast":
                    lower_arg = args[0].lower()
                    split_idx = lower_arg.rfind(" as ")
                    if split_idx != -1:
                        return self._extract_column_name(args[0][:split_idx])
                return self._extract_column_name(args[0])

        cleaned = self._cleanup_token(expression)
        return cleaned.split(".")[-1]

    def _normalize_constant(self, expression: str) -> Optional[str]:
        expression = self._normalize_predicate(expression)
        if not expression:
            return None

        cleaned = self._cleanup_token(expression)
        cleaned = self._remove_all_brackets(cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned or None

    def _qualify_predicates(
        self,
        table: Optional[str],
        predicates: List[PredicateTriple],
    ) -> List[PredicateTriple]:
        if not table:
            return predicates

        qualified: List[PredicateTriple] = []
        for predicate in predicates:
            if not predicate or len(predicate) < 2:
                continue

            column, operator, value = predicate
            if operator != "raw" and column and "." not in column:
                column = f"{table}.{column}"
            qualified.append([column, operator, value])

        return qualified

    def _split_top_level_items(self, text: str, delimiter: str) -> List[str]:
        items: List[str] = []
        start = 0
        paren_depth = 0
        bracket_depth = 0
        idx = 0

        while idx < len(text):
            char = text[idx]
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth -= 1
            elif char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth -= 1

            is_delimiter = False
            next_idx = idx + 1
            if paren_depth == 0 and bracket_depth == 0:
                if delimiter == "," and char == ",":
                    is_delimiter = True
                elif delimiter in {"AND", "OR"} and text.startswith(delimiter, idx):
                    prev_char = text[idx - 1] if idx > 0 else " "
                    delim_len = len(delimiter)
                    next_char = text[idx + delim_len] if idx + delim_len < len(text) else " "
                    if prev_char.isspace() and next_char.isspace():
                        is_delimiter = True
                        next_idx = idx + delim_len

            if is_delimiter:
                item = text[start:idx].strip()
                if item:
                    items.append(item)
                start = next_idx
                idx = next_idx
                continue

            idx += 1

        item = text[start:].strip()
        if item:
            items.append(item)
        return items

    def _strip_outer_parentheses(self, expression: str) -> str:
        expression = expression.strip()
        while expression.startswith("(") and expression.endswith(")"):
            depth = 0
            wraps_whole_expression = True
            for idx, char in enumerate(expression):
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                if depth == 0 and idx < len(expression) - 1:
                    wraps_whole_expression = False
                    break
            if not wraps_whole_expression:
                break
            expression = expression[1:-1].strip()
        return expression

    def _strip_outer_brackets(self, expression: str) -> str:
        expression = expression.strip()
        while expression.startswith("[") and expression.endswith("]"):
            depth = 0
            wraps_whole_expression = True
            for idx, char in enumerate(expression):
                if char == "[":
                    depth += 1
                elif char == "]":
                    depth -= 1
                if depth == 0 and idx < len(expression) - 1:
                    wraps_whole_expression = False
                    break
            if not wraps_whole_expression:
                break
            expression = expression[1:-1].strip()
        return expression

    def _normalize_predicate(self, predicate: str) -> str:
        predicate = self._strip_outer_parentheses(predicate)
        predicate = self._normalize_identifier(predicate)
        return re.sub(r"\s+", " ", predicate).strip()

    def _normalize_operator(self, operator: str) -> str:
        normalized = operator.lower().strip()
        return FUNCTION_OPERATOR_MAP.get(
            normalized, COMPARISON_OPERATOR_MAP.get(normalized, normalized)
        )

    def _cleanup_token(self, text: str) -> str:
        cleaned = self._normalize_identifier(text).strip()
        while True:
            next_cleaned = self._strip_outer_parentheses(cleaned)
            if next_cleaned != cleaned:
                cleaned = next_cleaned
                continue

            next_cleaned = self._strip_outer_brackets(cleaned)
            if next_cleaned != cleaned:
                cleaned = next_cleaned
                continue
            break

        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
            cleaned = cleaned[1:-1].strip()

        return re.sub(r"\s+", " ", cleaned).strip()

    def _remove_all_brackets(self, text: str) -> str:
        return text.translate(str.maketrans("", "", "()[]"))

    def _normalize_identifier(self, text: str) -> str:
        return re.sub(r"#\d+", "", text).strip()

    def _dedupe_keep_order(self, values: List[str]) -> List[str]:
        return list(dict.fromkeys(value for value in values if value))

    def _dedupe_predicate_triples(self, values: List[PredicateTriple]) -> List[PredicateTriple]:
        deduped: List[PredicateTriple] = []
        seen = set()
        for value in values:
            key = tuple(value)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(value)
        return deduped

    def _extract_data_filters(self, line: str) -> List[PredicateTriple]:
        filters = self._extract_bracket_section(line, "DataFilters:")
        if filters is None or "..." in filters:
            return []
        return self._structure_predicates(self._split_top_level_items(filters, ","))

    def _extract_bracket_section(self, text: str, token: str) -> Optional[str]:
        token_idx = text.find(token)
        if token_idx == -1:
            return None

        start_idx = text.find("[", token_idx + len(token))
        if start_idx == -1:
            return None

        depth = 0
        for idx in range(start_idx, len(text)):
            char = text[idx]
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    return text[start_idx + 1 : idx]
        return None

    def _fill_join_tables_bottom_up(self, tree: List[Node], idx: Optional[int]):
        if idx is None:
            return []
        node = tree[idx]
        if node.lc is not None:
            node.tables = list(dict.fromkeys(node.tables + self._fill_join_tables_bottom_up(tree, node.lc)))
        if node.rc is not None:
            node.tables = list(dict.fromkeys(node.tables + self._fill_join_tables_bottom_up(tree, node.rc)))
        return node.tables

    def _fill_root_stats(self, tree: List[Node], plan_info: PlanInfo):
        if len(tree) <= 1:
            return
        tree[1].card = self._safe_log_stat(plan_info.card)
        tree[1].size_in_bytes = self._safe_log_stat(plan_info.size)

    def print_tree(self, tree: List[Node], i: Optional[int] = 1, depth: int = 0) -> None:
        if i is None:
            return
        indent = "    " * depth
        logger.info(
            f"{indent}{tree[i].operator} {tree[i].executed} "
            f"{tree[i].tables} {tree[i].data} "
            f"{tree[i].card} {tree[i].size_in_bytes}"
        )
        self.print_tree(tree, tree[i].lc, depth + 1)
        self.print_tree(tree, tree[i].rc, depth + 1)


if __name__ == "__main__":
    import torch

    data = torch.load("/home/hejiahao/Calibra/data/l4k4rm3/JOB_20it.pt")
    pp = data[0]["x"]["plan_info"]
    pep = data[2]["x"]["plan_info"]

    processor = SparkPlanPreprocessor()
    pp_tree = processor.plan2tree(pp)
    processor.print_tree(pp_tree)
    pep_tree = processor.plan2tree(pep)
    processor.print_tree(pep_tree)
