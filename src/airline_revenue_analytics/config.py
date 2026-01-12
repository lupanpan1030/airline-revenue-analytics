"""Central config: repo paths and dataset location. (中央配置：仓库路径与数据集位置)"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Paths:
    """Resolved path bundle for a pipeline. (管线的路径集合)"""
    repo_root: Path
    data_raw: Path
    db_path: Path
    outputs_root: Path
    figures: Path
    tables: Path
    artifacts: Path


def repo_root() -> Path:
    """Return repository root by searching for pyproject.toml. (通过查找 pyproject.toml 来定位仓库根目录)"""
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


def get_db_path() -> Path:
    """Resolve SQLite DB path from env or default data/raw location. (从环境变量或 data/raw 默认位置解析 SQLite 路径)"""
    env = os.getenv("AIRLINE_DB_PATH")
    if env:
        env_path = Path(env).expanduser()
        if not env_path.is_file():
            raise FileNotFoundError(
                "AIRLINE_DB_PATH is set but no file exists at:\n"
                f"  {env_path}\n"
                "Fix the path or run `python scripts/fetch_data.py` to download "
                "datalearn/airlines-db into data/raw/airlines_db.sqlite."
            )
        return env_path

    default = repo_root() / "data" / "raw" / "airlines_db.sqlite"
    if default.is_file():
        return default

    raise FileNotFoundError(
        "SQLite DB not found at data/raw/airlines_db.sqlite.\n"
        "Run `python scripts/fetch_data.py` to download from Kaggle "
        "(datalearn/airlines-db), or set AIRLINE_DB_PATH to your local sqlite file."
    )


def get_paths(pipeline: str, create: bool = True) -> Paths:
    """Return standardized paths for a pipeline (booking or segment). (返回管线的标准路径集合)"""
    pipeline = pipeline.strip().lower()
    if pipeline not in {"booking", "segment"}:
        raise ValueError("pipeline must be 'booking' or 'segment'")

    root = repo_root()
    data_raw = root / "data" / "raw"
    outputs_root = root / "outputs" / pipeline
    figures = outputs_root / "figures"
    tables = outputs_root / "tables"
    artifacts = outputs_root / "artifacts"

    if create:
        # Ensure standard folders exist for outputs and data / 确保输出与数据目录存在
        for d in (data_raw, outputs_root, figures, tables, artifacts):
            d.mkdir(parents=True, exist_ok=True)

    return Paths(
        repo_root=root,
        data_raw=data_raw,
        db_path=get_db_path(),
        outputs_root=outputs_root,
        figures=figures,
        tables=tables,
        artifacts=artifacts,
    )
