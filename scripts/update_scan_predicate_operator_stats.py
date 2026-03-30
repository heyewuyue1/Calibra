from collections import Counter
from datetime import date
from pathlib import Path
import json
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocessor.sparkplanpreprocessor import SparkPlanPreprocessor


DATASETS = [
    "data/JOB_leap.pt",
    "data/STACK_leap.pt",
    "data/TPC-H_leap.pt",
]
IGNORED_OPERATORS = {"not", "isnull"}
EXAMPLE_OPERATORS = {"=", "<", ">", "contains"}
MAX_EXAMPLES_PER_DATASET = 12
MAX_EXAMPLES_PER_OPERATOR = 8
MAX_UNIQUE_EXAMPLES = 20
OUTPUT_PATH = ROOT / "data/scan_predicate_operator_stats.json"


def ordered_counts(counter: Counter) -> dict:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def main() -> None:
    processor = SparkPlanPreprocessor()
    result = {
        "generated_at": str(date.today()),
        "ignored_operators": sorted(IGNORED_OPERATORS),
        "datasets": {},
        "overall": {},
        "raw_analysis": {
            "unique_examples": [],
            "by_dataset": {},
            "category_summary": [],
            "category_top_examples": {},
        },
    }

    overall = Counter()
    category_examples = {operator: [] for operator in sorted(EXAMPLE_OPERATORS)}
    category_seen = {operator: set() for operator in EXAMPLE_OPERATORS}
    unique_examples = []
    unique_seen = set()

    for relative_path in DATASETS:
        path = ROOT / relative_path
        data = torch.load(path)
        operators = Counter()
        scans = 0
        predicates_excluding_not = 0
        examples = []
        example_seen = set()

        for sample in data:
            tree = processor.plan2tree(sample["x"]["plan_info"])
            query_id = sample["x"].get("query_id")

            for node in tree[1:]:
                if node.operator != "Scan":
                    continue

                scans += 1
                for predicate in node.data.get("predicates", []):
                    if not predicate or len(predicate) < 2:
                        continue

                    operator = predicate[1]
                    if operator in IGNORED_OPERATORS:
                        continue

                    operators[operator] += 1
                    predicates_excluding_not += 1

                    if operator not in EXAMPLE_OPERATORS:
                        continue

                    predicate_tuple = tuple(predicate)
                    sample_key = (relative_path, query_id, predicate_tuple)
                    example_entry = {
                        "query_id": query_id,
                        "predicate": list(predicate_tuple),
                    }

                    if sample_key not in example_seen and len(examples) < MAX_EXAMPLES_PER_DATASET:
                        example_seen.add(sample_key)
                        examples.append(example_entry)

                    if sample_key not in unique_seen and len(unique_examples) < MAX_UNIQUE_EXAMPLES:
                        unique_seen.add(sample_key)
                        unique_examples.append({
                            "dataset": relative_path,
                            **example_entry,
                        })

                    if (
                        sample_key not in category_seen[operator]
                        and len(category_examples[operator]) < MAX_EXAMPLES_PER_OPERATOR
                    ):
                        category_seen[operator].add(sample_key)
                        category_examples[operator].append({
                            "dataset": relative_path,
                            **example_entry,
                        })

        overall.update(operators)
        result["datasets"][relative_path] = {
            "samples": len(data),
            "scans": scans,
            "predicates_excluding_not": predicates_excluding_not,
            "operators": ordered_counts(operators),
        }
        result["raw_analysis"]["by_dataset"][relative_path] = examples

    result["overall"] = ordered_counts(overall)
    result["raw_analysis"]["unique_examples"] = unique_examples
    result["raw_analysis"]["category_summary"] = [
        {"operator": operator, "count": overall.get(operator, 0)}
        for operator in sorted(EXAMPLE_OPERATORS)
    ]
    result["raw_analysis"]["category_top_examples"] = category_examples

    with OUTPUT_PATH.open("w", encoding="utf-8") as file_obj:
        json.dump(result, file_obj, indent=2, ensure_ascii=False)
        file_obj.write("\n")


if __name__ == "__main__":
    main()
