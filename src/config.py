from dataclasses import dataclass
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import re


PROJECT_ROOT = Path("/home/hejiahao/Calibra")
PREFIX = str(PROJECT_ROOT)
PROJECT_PARENT = str(PROJECT_ROOT.parent)
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_BENCHMARK = "STACK"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class BenchmarkRunSpec:
    benchmark: str
    database: str
    table_key: str
    model_path: str
    result_label: str

    @property
    def baseline_cbo_path(self):
        return f"{PREFIX}/results/baselines/{self.benchmark}_cbo.json"

    @property
    def baseline_leap_path(self):
        return f"{PREFIX}/results/baselines/{self.benchmark}_leap.json"

    @property
    def result_path(self):
        return f"{PREFIX}/results/{self.result_label}.json"


PRETRAINED_BENCHMARK_SPECS = {
    "JOB": BenchmarkRunSpec(
        benchmark="JOB",
        database="imdb_10x",
        table_key="imdb",
        model_path=f"{PREFIX}/data/l4k4rm3/models/JOB_20its_predicate_optimized.pt",
        result_label="JOB_20its",
    ),
    "STACK": BenchmarkRunSpec(
        benchmark="STACK",
        database="stack",
        table_key="stack",
        model_path=f"{PREFIX}/data/STACK/models/STACK_40its_predicate_optimized.pt",
        result_label="STACK_40its",
    ),
    "TPC-H": BenchmarkRunSpec(
        benchmark="TPC-H",
        database="tpch_sf100",
        table_key="tpch",
        model_path=f"{PREFIX}/data/TPC-H/models/TPC-H_40its_predicate_optimized.pt",
        result_label="TPC-H_40its",
    ),
}

BENCHMARK_ALIASES = {
    "TPCH": "TPC-H",
}

TABLE_DIMENSIONS = {
    "tpch": {"table_num": 8, "col_num": 16},
    "imdb": {"table_num": 17, "col_num": 34},
    "stack": {"table_num": 11, "col_num": 25},
    "tpcds": {"table_num": 25, "col_num": 0},
}


def normalize_benchmark_name(benchmark):
    normalized = (benchmark or DEFAULT_BENCHMARK).upper()
    return BENCHMARK_ALIASES.get(normalized, normalized)


def get_pretrained_run_spec(benchmark):
    normalized = normalize_benchmark_name(benchmark)
    if normalized not in PRETRAINED_BENCHMARK_SPECS:
        supported = ", ".join(sorted(PRETRAINED_BENCHMARK_SPECS))
        raise ValueError(f"Unsupported benchmark: {benchmark}. Expected one of: {supported}")
    return PRETRAINED_BENCHMARK_SPECS[normalized]


def sanitize_run_id(run_id):
    if not run_id:
        raise ValueError("run_id must be a non-empty string")
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError(
            "run_id may contain only letters, digits, dots, underscores, and dashes"
        )
    return run_id


def resolve_run_id(run_id=None):
    candidate = run_id or os.getenv("CALIBRA_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
    return sanitize_run_id(candidate)


@dataclass(frozen=True)
class RunArtifacts:
    benchmark: str
    run_id: str
    predicate_encoding: bool = True

    def __post_init__(self):
        object.__setattr__(self, "benchmark", normalize_benchmark_name(self.benchmark))
        object.__setattr__(self, "run_id", sanitize_run_id(self.run_id))

    @property
    def root_dir(self):
        return ARTIFACTS_ROOT / self.benchmark / "runs" / self.run_id

    @property
    def dataset_dir(self):
        return self.root_dir / "dataset"

    @property
    def model_dir(self):
        return self.root_dir / "model"

    @property
    def train_dir(self):
        return self.root_dir / "train"

    @property
    def eval_dir(self):
        return self.root_dir / "eval"

    @property
    def plots_dir(self):
        return self.root_dir / "plots"

    @property
    def conf_dir(self):
        return self.root_dir / "conf"

    @property
    def log_dir(self):
        return self.root_dir / "log"

    @property
    def raw_training_data_path(self):
        return str(self.dataset_dir / "raw_train.pt")

    @property
    def merged_training_data_path(self):
        return str(self.dataset_dir / "merged_train.pt")

    @property
    def bootstrap_data_path(self):
        return str(self.dataset_dir / "bootstrap_data.pt")

    @property
    def bootstrap_samples_path(self):
        return str(self.dataset_dir / "bootstrap_samples.pt")

    @property
    def model_path(self):
        return str(self.model_dir / "model.pt")

    @property
    def bootstrap_model_path(self):
        return str(self.model_dir / "bootstrap_model.pt")

    @property
    def metrics_path(self):
        return str(self.train_dir / "metrics.csv")

    @property
    def pointwise_tensorboard_dir(self):
        return str(self.train_dir / "tensorboard" / "pointwise")

    @property
    def bootstrap_tensorboard_dir(self):
        return str(self.train_dir / "tensorboard" / "bootstrap")

    @property
    def bootstrap_loss_plot_path(self):
        return str(self.train_dir / "bootstrap_loss.png")

    @property
    def latency_path(self):
        return str(self.eval_dir / "latency.json")

    @property
    def manifest_path(self):
        return str(self.root_dir / "manifest.json")

    def conf_artifact_path(self, name):
        return str(self.conf_dir / name)

    def log_artifact_path(self, name):
        return str(self.log_dir / name)

    def comparison_plot_path(self, left_label, right_label="model"):
        return str(self.plots_dir / f"{left_label}_vs_{right_label}.png")

    def manifest_defaults(self):
        return {
            "benchmark": self.benchmark,
            "description": "",
            "run_id": self.run_id,
            "paths": {
                "root_dir": str(self.root_dir),
                "conf_dir": str(self.conf_dir),
                "log_dir": str(self.log_dir),
                "raw_training_data_path": self.raw_training_data_path,
                "model_path": self.model_path,
                "metrics_path": self.metrics_path,
                "pointwise_tensorboard_dir": self.pointwise_tensorboard_dir,
                "latency_path": self.latency_path,
                "manifest_path": self.manifest_path,
                "plots_dir": str(self.plots_dir),
            },
        }


def get_run_artifacts(benchmark=None, run_id=None, predicate_encoding=True):
    return RunArtifacts(
        benchmark=normalize_benchmark_name(benchmark),
        run_id=resolve_run_id(run_id),
        predicate_encoding=predicate_encoding,
    )


def ensure_parent_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _merge_dict(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_dict(target[key], value)
        else:
            target[key] = value


def update_manifest(manifest_path, updates):
    manifest = {}
    path = Path(manifest_path)
    if path.exists():
        with path.open("r") as f:
            manifest = json.load(f)

    normalized_updates = _to_jsonable(updates)
    if manifest.get("description") and normalized_updates.get("description") == "":
        normalized_updates.pop("description")

    _merge_dict(manifest, normalized_updates)
    ensure_parent_dir(path)
    with path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return str(path)


DEFAULT_CONFIG_RUN_ID = resolve_run_id()


class ServerConfig:
    raw_data_path = get_run_artifacts(DEFAULT_BENCHMARK, DEFAULT_CONFIG_RUN_ID).raw_training_data_path
    bootstrap_data_path = get_run_artifacts(DEFAULT_BENCHMARK, DEFAULT_CONFIG_RUN_ID).bootstrap_data_path
    data_path = f"{PREFIX}/data/STACK/STACK_40it.pt"


class TestConfig:
    def __init__(
        self,
        method,
        database,
        benchmark,
        max_retry,
        repeats,
        save_latency,
        save_path=None,
        conf_path=None,
        run_id=None,
        sqldir=None,
    ) -> None:
        self.method = method
        self.database = database
        self.benchmark = normalize_benchmark_name(benchmark)
        self.timeout = 300
        self.working_dir = PROJECT_PARENT
        self.max_retry = max_retry
        self.repeats = repeats
        self.explain_only = True if method == "bootstrap" else False
        self.conf_path = conf_path or f"{PREFIX}/conf/{method}.conf"
        benchmark_root = (PROJECT_ROOT / "benchmark").resolve()
        if sqldir is None:
            self.benchmark_path = f"{PREFIX}/benchmark/{self.benchmark}/"
        else:
            resolved_sqldir = Path(sqldir)
            if not resolved_sqldir.is_absolute():
                resolved_sqldir = benchmark_root / resolved_sqldir
            resolved_sqldir = resolved_sqldir.resolve()
            if resolved_sqldir != benchmark_root and benchmark_root not in resolved_sqldir.parents:
                raise ValueError(f"sqldir must be inside {benchmark_root}: {sqldir}")
            if not resolved_sqldir.is_dir():
                raise ValueError(f"sqldir does not exist: {sqldir}")
            self.benchmark_path = str(resolved_sqldir)
        self.save_latency = save_latency
        self.artifacts = get_run_artifacts(self.benchmark, run_id)
        self.run_id = self.artifacts.run_id
        self.save_path = save_path or self.artifacts.latency_path


class TrainConfig:
    inference_only = False
    patience = 10
    epochs = 100
    batch_size = 128
    bootstrap_sample_size = 2000
    save_bootstrap_samples = False
    current_time = DEFAULT_CONFIG_RUN_ID
    enable_predicate_encoding = True

    @classmethod
    def feature_variant_suffix(cls):
        return "_predicate_optimized" if cls.enable_predicate_encoding else ""

    @classmethod
    def _default_artifacts(cls):
        return get_run_artifacts(
            DEFAULT_BENCHMARK,
            cls.current_time,
            predicate_encoding=cls.enable_predicate_encoding,
        )

    @classmethod
    def bootstrap_samples_save_path(cls):
        return cls._default_artifacts().bootstrap_samples_path

    @classmethod
    def log_save_path(cls):
        return str(Path(cls._default_artifacts().train_dir) / "training")

    @classmethod
    def model_save_path(cls):
        return cls._default_artifacts().model_path

    @classmethod
    def bootstrap_model_save_path(cls):
        return cls._default_artifacts().bootstrap_model_path


class EnvironmentConfig:
    database = "stack"
    table_key = "stack"
    table_file = f"{PREFIX}/benchmark/stack_tables.csv"
    table_num = 11
    col_num = 25

    @classmethod
    def configure(cls, database, table_key):
        dimensions = TABLE_DIMENSIONS.get(table_key)
        if dimensions is None:
            supported = ", ".join(sorted(TABLE_DIMENSIONS))
            raise ValueError(f"Unsupported table key: {table_key}. Expected one of: {supported}")

        cls.database = database
        cls.table_key = table_key
        cls.table_file = f"{PREFIX}/benchmark/{table_key}_tables.csv"
        cls.table_num = dimensions["table_num"]
        cls.col_num = dimensions.get("col_num", 0)

    @classmethod
    def configure_for_benchmark(cls, benchmark, database=None, table_key=None):
        spec = get_pretrained_run_spec(benchmark)
        cls.configure(database or spec.database, table_key or spec.table_key)


class LoggingConfig:
    log_level = logging.INFO
