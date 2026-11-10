from dataclasses import dataclass, field, fields

import yaml

from ccr.corruptions import CORRUPTIONS, SEVERITIES
from ccr.features import validate_spec
from ccr.models import MODELS

DATASETS = ("cifar10", "digits")


@dataclass
class Regime:
    corruptions: list = field(default_factory=list)  # empty means plain clean training
    severities: list = field(default_factory=lambda: [1, 2, 3])
    p: float = 0.5


@dataclass
class Config:
    dataset: str
    data_root: str = "data"
    n_train: int = 10000
    n_test: int = 2000
    seeds: list = field(default_factory=lambda: [0])
    features: list = field(default_factory=lambda: ["hog"])
    models: list = field(default_factory=lambda: ["logreg"])
    corruptions: list = field(default_factory=lambda: list(CORRUPTIONS))
    severities: list = field(default_factory=lambda: list(SEVERITIES))
    regimes: dict = field(default_factory=lambda: {"clean": Regime()})


def _corruption_list(value, exclude=()):
    if isinstance(value, str):
        if value != "all":
            raise ValueError(f'corruptions must be "all" or a list, got {value!r}')
        names = list(CORRUPTIONS)
    else:
        names = list(value)
    for name in [*names, *exclude]:
        if name not in CORRUPTIONS:
            raise ValueError(f"unknown corruption {name!r}, choose from {sorted(CORRUPTIONS)}")
    return [n for n in names if n not in exclude]


def _severity_list(value, where):
    levels = sorted(set(value))
    if not levels or any(s not in SEVERITIES for s in levels):
        raise ValueError(f"{where}: severities must be a non-empty subset of {list(SEVERITIES)}")
    return levels


def _parse_regime(name, raw):
    if not raw:
        return Regime()
    if not isinstance(raw, dict):
        raise ValueError(f"regime {name!r} must be a mapping")
    unknown = set(raw) - {"corruptions", "exclude", "severities", "p"}
    if unknown:
        raise ValueError(f"regime {name!r}: unknown keys {sorted(unknown)}")
    names = _corruption_list(raw.get("corruptions", "all"), raw.get("exclude", []))
    if not names:
        raise ValueError(f"regime {name!r} has no corruptions left to train on")
    levels = _severity_list(raw.get("severities", [1, 2, 3]), f"regime {name!r}")
    p = float(raw.get("p", 0.5))
    if not 0 < p <= 1:
        raise ValueError(f"regime {name!r}: p must be in (0, 1]")
    return Regime(names, levels, p)


def _check_names(label, values, valid=None):
    if not values:
        raise ValueError(f"{label} must not be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} contains duplicates")
    if valid is not None:
        for v in values:
            if v not in valid:
                raise ValueError(f"unknown {label[:-1]} {v!r}, choose from {sorted(valid)}")


def parse_config(raw):
    if not isinstance(raw, dict):
        raise ValueError("config must be a mapping")
    unknown = set(raw) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    if "dataset" not in raw:
        raise ValueError("config needs a 'dataset' key")

    plain = {k: v for k, v in raw.items() if k not in ("corruptions", "severities", "regimes")}
    cfg = Config(**plain)
    cfg.corruptions = _corruption_list(raw.get("corruptions", "all"))
    cfg.severities = _severity_list(raw.get("severities", list(SEVERITIES)), "top level")
    cfg.regimes = {str(k): _parse_regime(k, v) for k, v in raw.get("regimes", {"clean": {}}).items()}

    if cfg.dataset not in DATASETS:
        raise ValueError(f"unknown dataset {cfg.dataset!r}, choose from {list(DATASETS)}")
    for key in ("n_train", "n_test"):
        v = getattr(cfg, key)
        if not isinstance(v, int) or v <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not cfg.corruptions:
        raise ValueError("corruptions must not be empty")
    if not cfg.regimes:
        raise ValueError("regimes must not be empty")
    _check_names("seeds", cfg.seeds)
    if any(not isinstance(s, int) or s < 0 for s in cfg.seeds):
        raise ValueError("seeds must be non-negative integers")
    _check_names("features", cfg.features)
    for spec in cfg.features:
        validate_spec(spec)
    _check_names("models", cfg.models, MODELS)
    return cfg


def load_config(path):
    with open(path) as f:
        return parse_config(yaml.safe_load(f))
