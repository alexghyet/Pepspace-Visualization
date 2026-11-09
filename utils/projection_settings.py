"""Shared deterministic projection settings for the app and precomputation."""

import json
import math
from numbers import Integral
from pathlib import Path


DEFAULT_NEIGHBORS = 5
DEFAULT_MIN_DIST = 0.005527128631029834
DEFAULT_METRIC = "cosine"
DEFAULT_SEED = 42


def _integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    try:
        converted = int(value)
        if float(value) != converted:
            raise ValueError
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be an integer.") from error
    return converted


def _number(value, name, *, minimum, maximum=None, strict_minimum=False):
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number.") from error
    valid_minimum = converted > minimum if strict_minimum else converted >= minimum
    if not math.isfinite(converted) or not valid_minimum or (maximum is not None and converted > maximum):
        operator = ">" if strict_minimum else ">="
        limit = f" and <= {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be finite, {operator} {minimum}{limit}.")
    return converted


def _method(value):
    token = str(value).strip().split()
    methods = {"umap": "UMAP", "densmap": "densMAP"}
    if not token or token[0].casefold() not in methods:
        raise ValueError("Projection method must be UMAP or densMAP.")
    return methods[token[0].casefold()]


def load_optuna_parameters(path):
    """Read an optional optimization file with an actionable malformed-file error."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        parameters = document["best_params"]
        if not isinstance(parameters, dict):
            raise ValueError("best_params must be an object")
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Could not read projection settings from {path.name}: {error}") from error
    return parameters


def projection_defaults(sample_count, best=None):
    """Create widget defaults using the same limits and values as the CLI."""
    if not isinstance(sample_count, Integral) or isinstance(sample_count, bool) or sample_count < 3:
        raise ValueError("At least three samples are required for projection settings.")
    best = {} if best is None else best
    neighbors = _integer(best.get("n_neighbors", DEFAULT_NEIGHBORS), "n_neighbors")
    return {
        "n_neighbors": min(200, sample_count - 1, max(2, neighbors)),
        "min_dist": _number(best.get("min_dist", DEFAULT_MIN_DIST), "min_dist", minimum=0, maximum=1),
        "method": _method(best.get("method", "UMAP")),
        "dens_lambda": _number(best.get("dens_lambda", 2.0), "dens_lambda", minimum=0),
        "dens_frac": _number(best.get("dens_frac", 0.3), "dens_frac", minimum=0, maximum=1),
        "dens_var_shift": _number(best.get("dens_var_shift", 0.1), "dens_var_shift", minimum=0, strict_minimum=True),
    }


def projection_arguments(settings, *, metric=DEFAULT_METRIC, seed=DEFAULT_SEED):
    """Translate applied widget settings into the cached runtime's arguments."""
    neighbors = _integer(settings["n_neighbors"], "n_neighbors")
    if neighbors < 2 or neighbors > 200:
        raise ValueError("n_neighbors must be between 2 and 200.")
    method = _method(settings["method"])
    if not isinstance(metric, str) or not metric.strip():
        raise ValueError("A projection metric is required.")
    return {
        "n_neighbors": neighbors,
        "min_dist": _number(settings["min_dist"], "min_dist", minimum=0, maximum=1),
        "metric": metric,
        "seed": _integer(seed, "seed"),
        "method": method,
        "dens_lambda": _number(settings.get("dens_lambda", 2.0), "dens_lambda", minimum=0) if method == "densMAP" else None,
        "dens_frac": _number(settings.get("dens_frac", 0.3), "dens_frac", minimum=0, maximum=1) if method == "densMAP" else None,
        "dens_var_shift": _number(settings.get("dens_var_shift", 0.1), "dens_var_shift", minimum=0, strict_minimum=True) if method == "densMAP" else None,
    }
