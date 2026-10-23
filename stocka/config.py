"""Demo settings and example-screen thresholds, loaded from TOML and validated.

The thresholds belong to the illustrative screen in `stocka.screen`. The
engine's own rules (vetos in `dq_score`, position limits, stops, drawdown
tiers) are fixed in code and cannot be configured here.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Invalid or unreadable configuration."""


def _check_int(name: str, value: Any, *, low: int, high: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer, got {value!r}")
    if value < low or (high is not None and value > high):
        bounds = f">= {low}" if high is None else f"between {low} and {high}"
        raise ConfigError(f"{name} must be {bounds}, got {value}")


def _check_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _check_bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be true or false, got {value!r}")


@dataclass(frozen=True)
class DemoSettings:
    seed: int = 1
    tickers: int = 20
    bootstrap_sims: int = 2000

    def __post_init__(self) -> None:
        _check_int("demo.seed", self.seed, low=0)
        _check_int("demo.tickers", self.tickers, low=1, high=99)
        _check_int("demo.bootstrap_sims", self.bootstrap_sims, low=0, high=20_000)
        if 0 < self.bootstrap_sims < 100:
            raise ConfigError("demo.bootstrap_sims must be 0 (off) or at least 100")


@dataclass(frozen=True)
class ScreenSettings:
    min_dq_score: int = 70
    max_beneish_m: float = -2.22
    exclude_sloan_q5: bool = True
    require_verified_audit: bool = True

    def __post_init__(self) -> None:
        _check_int("screen.min_dq_score", self.min_dq_score, low=0, high=100)
        object.__setattr__(self, "max_beneish_m", _check_float("screen.max_beneish_m", self.max_beneish_m))
        _check_bool("screen.exclude_sloan_q5", self.exclude_sloan_q5)
        _check_bool("screen.require_verified_audit", self.require_verified_audit)


@dataclass(frozen=True)
class Config:
    demo: DemoSettings = field(default_factory=DemoSettings)
    screen: ScreenSettings = field(default_factory=ScreenSettings)

    def with_overrides(self, *, seed: int | None = None, tickers: int | None = None) -> Config:
        changes = {k: v for k, v in (("seed", seed), ("tickers", tickers)) if v is not None}
        return replace(self, demo=replace(self.demo, **changes)) if changes else self


def _section(cls: type, name: str, raw: Any) -> Any:
    if not isinstance(raw, dict):
        raise ConfigError(f"[{name}] must be a table")
    unknown = sorted(set(raw) - {f.name for f in fields(cls)})
    if unknown:
        raise ConfigError(f"unknown key(s) in [{name}]: {', '.join(unknown)}")
    return cls(**raw)


def config_from_dict(data: dict) -> Config:
    unknown = sorted(set(data) - {"demo", "screen"})
    if unknown:
        raise ConfigError(f"unknown section(s): {', '.join(unknown)}")
    return Config(
        demo=_section(DemoSettings, "demo", data.get("demo", {})),
        screen=_section(ScreenSettings, "screen", data.get("screen", {})),
    )


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        return Config()
    path = Path(path)
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from None
    return config_from_dict(data)
