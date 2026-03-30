from collections import Counter
from pathlib import Path
import json
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocessor.sparkplanpreprocessor import SparkPlanPreprocessor


OUTPUT_DIR = ROOT / "data/examples"

EXAMPLES = [
    {
        "slug": "01_job_29a_deep_join_with_stages",
        "dataset": "data/JOB_leap.pt",
        "sample_index": 531,
        "query_id": "1773145162213-29a.sql",
        "title": "JOB deep join tree with rich predicates and query stages",
        "reason": "Very deep join tree with equality, range, contains, and multiple query stages.",
    },
    {
        "slug": "02_stack_q11_inset_expansion",
        "dataset": "data/STACK_leap.pt",
        "sample_index": 36,
        "query_id": "1774252740191-q11#0.sql",
        "title": "STACK example showing INSET expansion into multiple equality predicates",
        "reason": "Compact example that clearly shows INSET becoming many '=' predicates plus range filters.",
    },
    {
        "slug": "03_stack_q3_complex_join_with_stages",
        "dataset": "data/STACK_leap.pt",
        "sample_index": 619,
        "query_id": "1774255011711-q3#3.sql",
        "title": "STACK multi-join example with query stages",
        "reason": "High-join stack query with a non-empty queryStages map and processed scan predicates.",
    },
    {
        "slug": "04_tpch_q10_range_with_stages",
        "dataset": "data/TPC-H_leap.pt",
        "sample_index": 1,
        "query_id": "1773399927779-q10#0.sql",
        "title": "TPC-H range-predicate example with query stages",
        "reason": "Clear date-range example showing both '>' and '<' alongside staged execution info.",
    },
    {
        "slug": "05_tpch_q2_multi_join_with_contains",
        "dataset": "data/TPC-H_leap.pt",
        "sample_index": 83,
        "query_id": "1773400643903-q2#1.sql",
        "title": "TPC-H multi-join example with contains predicates and stages",
        "reason": "Broader join structure with staged plans and a contains-style predicate.",
    },
]


def render_tree(tree, idx=1, depth=0):
    if idx is None:
        return []
    node = tree[idx]
    indent = "    " * depth
    lines = [
        f"{indent}{node.operator} {node.executed} "
        f"{node.tables} {node.data} {node.card} {node.size_in_bytes}"
    ]
    lines.extend(render_tree(tree, node.lc, depth + 1))
    lines.extend(render_tree(tree, node.rc, depth + 1))
    return lines


def summarize_tree(tree):
    operator_counts = Counter(node.operator for node in tree[1:])
    predicate_counts = Counter()
    scan_summaries = []

    for node in tree[1:]:
        if node.operator != "Scan":
            continue
        predicates = node.data.get("predicates", [])
        for predicate in predicates:
            if len(predicate) > 1:
                predicate_counts[predicate[1]] += 1
        scan_summaries.append(
            {
                "tables": node.tables,
                "columns": node.data.get("columns", []),
                "predicates": predicates,
            }
        )

    return {
        "node_count": len(tree) - 1,
        "join_count": sum(1 for node in tree[1:] if "Join" in node.operator),
        "scan_count": sum(1 for node in tree[1:] if node.operator == "Scan"),
        "logical_query_stage_count": sum(
            1 for node in tree[1:] if node.operator == "LogicalQueryStage"
        ),
        "operator_counts": dict(operator_counts),
        "predicate_operator_counts": dict(predicate_counts),
        "scan_summaries": scan_summaries,
    }


def format_query_stages(plan_info):
    if not plan_info.queryStages:
        return "None\n"

    sections = []
    for stage_name, stage_info in plan_info.queryStages.items():
        sections.append(
            "\n".join(
                [
                    f"Stage: {stage_name}",
                    f"materialized: {stage_info.materialized}",
                    f"card: {stage_info.card}",
                    f"size: {stage_info.size}",
                    "stagePlan:",
                    stage_info.stagePlan,
                ]
            )
        )
    return "\n\n".join(sections) + "\n"


def export_example(processor, config):
    dataset_path = ROOT / config["dataset"]
    sample = torch.load(dataset_path)[config["sample_index"]]
    query_id = sample["x"].get("query_id")
    if query_id != config["query_id"]:
        raise ValueError(
            f"Sample mismatch for {config['slug']}: "
            f"expected {config['query_id']}, got {query_id}"
        )

    plan_info = sample["x"]["plan_info"]
    tree = processor.plan2tree(plan_info)
    summary = summarize_tree(tree)
    content = "\n".join(
        [
            f"title: {config['title']}",
            f"dataset: {config['dataset']}",
            f"sample_index: {config['sample_index']}",
            f"query_id: {query_id}",
            f"reason: {config['reason']}",
            f"plan_card: {plan_info.card}",
            f"plan_size: {plan_info.size}",
            f"query_stage_count: {len(plan_info.queryStages)}",
            "",
            "== Processed Summary ==",
            json.dumps(summary, indent=2, ensure_ascii=False),
            "",
            "== Original PlanInfo.plan ==",
            plan_info.plan,
            "",
            "== Original PlanInfo.queryStages ==",
            format_query_stages(plan_info).rstrip(),
            "",
            "== sparkplanpreprocessor.print_tree ==",
            "\n".join(render_tree(tree)),
            "",
        ]
    )

    output_path = OUTPUT_DIR / f"{config['slug']}.txt"
    output_path.write_text(content, encoding="utf-8")
    return output_path


def build_index(paths_by_slug):
    lines = ["# Preprocessor Examples", ""]
    for config in EXAMPLES:
        path = paths_by_slug[config["slug"]]
        lines.extend(
            [
                f"## {config['title']}",
                f"- File: `{path.relative_to(ROOT)}`",
                f"- Dataset: `{config['dataset']}`",
                f"- Sample index: `{config['sample_index']}`",
                f"- Query ID: `{config['query_id']}`",
                f"- Reason: {config['reason']}",
                "",
            ]
        )
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    processor = SparkPlanPreprocessor()
    paths_by_slug = {}
    for config in EXAMPLES:
        output_path = export_example(processor, config)
        paths_by_slug[config["slug"]] = output_path
    build_index(paths_by_slug)


if __name__ == "__main__":
    main()
