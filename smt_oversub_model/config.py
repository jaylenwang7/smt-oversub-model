"""
Configuration loading and serialization for experiment configs.

Provides JSON-serializable config structures and conversion utilities.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Callable, List, Any, Dict
import json
from pathlib import Path

try:
    import json5
    _HAS_JSON5 = True
except ImportError:
    _HAS_JSON5 = False

from .model import polynomial_power_curve_fn


# --- Power Curve Presets ---

POWER_CURVE_PRESETS: Dict[str, Callable[[float], float]] = {
    "linear": lambda u: u,
    "specpower": lambda u: u ** 0.9,
    "polynomial": polynomial_power_curve_fn(),  # Default freq
}


def make_power_curve_fn(
    curve_type: str,
    exponent: Optional[float] = None,
    freq_mhz: Optional[float] = None,
) -> Callable[[float], float]:
    """
    Create a power curve function from a type name and optional parameters.

    Args:
        curve_type: One of "linear", "power", "specpower", or "polynomial"
        exponent: Required for "power" type, ignored for others
        freq_mhz: Optional frequency for "polynomial" type (default 3500 MHz)

    Returns:
        Callable mapping utilization [0,1] -> [0,1]
    """
    if curve_type == "linear":
        return lambda u: u
    elif curve_type == "specpower":
        return lambda u: u ** 0.9
    elif curve_type == "power":
        if exponent is None:
            raise ValueError("power curve type requires 'exponent' parameter")
        exp = exponent  # Capture in closure
        return lambda u: u ** exp
    elif curve_type == "polynomial":
        freq = freq_mhz if freq_mhz is not None else 3500.0
        return polynomial_power_curve_fn(freq_mhz=freq)
    else:
        raise ValueError(f"Unknown power curve type: {curve_type}. "
                        f"Valid types: linear, power, specpower, polynomial")


# --- Config Dataclasses ---

@dataclass
class PowerCurveSpec:
    """Serializable specification for a power curve."""
    type: str = "specpower"
    exponent: Optional[float] = None
    freq_mhz: Optional[float] = None  # For polynomial curve type

    def to_callable(self) -> Callable[[float], float]:
        """Convert spec to actual callable function."""
        return make_power_curve_fn(self.type, self.exponent, self.freq_mhz)

    def to_dict(self) -> dict:
        d = {"type": self.type}
        if self.exponent is not None:
            d["exponent"] = self.exponent
        if self.freq_mhz is not None:
            d["freq_mhz"] = self.freq_mhz
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "PowerCurveSpec":
        return cls(
            type=data.get("type", "specpower"),
            exponent=data.get("exponent"),
            freq_mhz=data.get("freq_mhz"),
        )


@dataclass
class ProcessorSpec:
    """Specification for a processor configuration.

    Args:
        physical_cores: Number of physical CPU cores
        threads_per_core: SMT threads per core (1 = no SMT, 2+ = SMT enabled)
        power_idle_w: Idle power consumption in watts
        power_max_w: Maximum power consumption in watts
        core_overhead: Number of pCPUs reserved for host (not available for VMs)
    """
    physical_cores: int = 48
    threads_per_core: int = 1  # 1 = no SMT, 2 = SMT (hyperthreading)
    power_idle_w: float = 100.0
    power_max_w: float = 400.0
    core_overhead: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ProcessorSpec":
        return cls(
            physical_cores=data.get("physical_cores", 48),
            threads_per_core=data.get("threads_per_core", 1),
            power_idle_w=data.get("power_idle_w", 100.0),
            power_max_w=data.get("power_max_w", 400.0),
            core_overhead=data.get("core_overhead", 0),
        )

    @property
    def is_smt(self) -> bool:
        """Return True if this processor has SMT enabled (threads_per_core > 1)."""
        return self.threads_per_core > 1


@dataclass
class ProcessorConfigSpec:
    """Container for named processor configurations.

    Processors are stored in a dict keyed by name, allowing arbitrary
    processor definitions (not limited to 'smt'/'nosmt').
    """
    processors: Dict[str, ProcessorSpec] = field(default_factory=dict)

    def __post_init__(self):
        # Ensure at least default processors exist for backward compatibility
        if not self.processors:
            self.processors = {
                "smt": ProcessorSpec(physical_cores=48, threads_per_core=2,
                                    power_idle_w=100.0, power_max_w=400.0),
                "nosmt": ProcessorSpec(physical_cores=48, threads_per_core=1,
                                      power_idle_w=90.0, power_max_w=340.0),
            }

    def to_dict(self) -> dict:
        return {name: spec.to_dict() for name, spec in self.processors.items()}

    @classmethod
    def from_dict(cls, data: dict) -> "ProcessorConfigSpec":
        processors = {}
        for name, spec_data in data.items():
            processors[name] = ProcessorSpec.from_dict(spec_data)
        return cls(processors=processors)

    def get(self, name: str) -> ProcessorSpec:
        """Get a processor by name."""
        if name not in self.processors:
            raise KeyError(f"Unknown processor: {name}. Available: {list(self.processors.keys())}")
        return self.processors[name]

    # Backward compatibility properties
    @property
    def smt(self) -> ProcessorSpec:
        """Get the 'smt' processor (backward compatibility)."""
        return self.get("smt")

    @property
    def nosmt(self) -> ProcessorSpec:
        """Get the 'nosmt' processor (backward compatibility)."""
        return self.get("nosmt")


@dataclass
class WorkloadSpec:
    """Workload specification."""
    total_vcpus: int = 10000
    avg_util: float = 0.3

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "WorkloadSpec":
        return cls(
            total_vcpus=data.get("total_vcpus", 10000),
            avg_util=data.get("avg_util", 0.3),
        )


@dataclass
class CostSpec:
    """Cost and carbon parameters."""
    embodied_carbon_kg: float = 1000.0
    server_cost_usd: float = 15000.0
    carbon_intensity_g_kwh: float = 400.0
    electricity_cost_usd_kwh: float = 0.10
    lifetime_years: float = 4.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CostSpec":
        return cls(
            embodied_carbon_kg=data.get("embodied_carbon_kg", 1000.0),
            server_cost_usd=data.get("server_cost_usd", 15000.0),
            carbon_intensity_g_kwh=data.get("carbon_intensity_g_kwh", 400.0),
            electricity_cost_usd_kwh=data.get("electricity_cost_usd_kwh", 0.10),
            lifetime_years=data.get("lifetime_years", 4.0),
        )


@dataclass
class OversubSpec:
    """Oversubscription configuration."""
    smt_ratio: float = 1.3
    smt_util_overhead: float = 0.05
    nosmt_util_overhead: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "OversubSpec":
        return cls(
            smt_ratio=data.get("smt_ratio", 1.3),
            smt_util_overhead=data.get("smt_util_overhead", 0.05),
            nosmt_util_overhead=data.get("nosmt_util_overhead", 0.0),
        )


@dataclass
class SweepSpec:
    """Specification for a parameter sweep."""
    parameter: str
    values: List[float]

    def to_dict(self) -> dict:
        return {
            "parameter": self.parameter,
            "values": self.values,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SweepSpec":
        return cls(
            parameter=data["parameter"],
            values=data["values"],
        )


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration.

    This is the top-level config that gets serialized to/from JSON.
    """
    name: str
    description: str = ""
    processor: ProcessorConfigSpec = field(default_factory=ProcessorConfigSpec)
    power_curve: PowerCurveSpec = field(default_factory=PowerCurveSpec)
    workload: WorkloadSpec = field(default_factory=WorkloadSpec)
    cost: CostSpec = field(default_factory=CostSpec)
    oversubscription: OversubSpec = field(default_factory=OversubSpec)
    sweep: Optional[SweepSpec] = None

    def to_dict(self) -> dict:
        """Convert config to JSON-serializable dict."""
        d = {
            "name": self.name,
            "description": self.description,
            "processor": self.processor.to_dict(),
            "power_curve": self.power_curve.to_dict(),
            "workload": self.workload.to_dict(),
            "cost": self.cost.to_dict(),
            "oversubscription": self.oversubscription.to_dict(),
        }
        if self.sweep is not None:
            d["sweep"] = self.sweep.to_dict()
        else:
            d["sweep"] = None
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        """Create config from dict (e.g., from JSON)."""
        sweep = None
        if data.get("sweep"):
            sweep = SweepSpec.from_dict(data["sweep"])

        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            processor=ProcessorConfigSpec.from_dict(data.get("processor", {})),
            power_curve=PowerCurveSpec.from_dict(data.get("power_curve", {})),
            workload=WorkloadSpec.from_dict(data.get("workload", {})),
            cost=CostSpec.from_dict(data.get("cost", {})),
            oversubscription=OversubSpec.from_dict(data.get("oversubscription", {})),
            sweep=sweep,
        )

    def is_sweep(self) -> bool:
        """Return True if this config specifies a sweep."""
        return self.sweep is not None

    def get_sweepable_parameters(self) -> List[str]:
        """Return list of parameters that can be swept."""
        return [
            "avg_util",
            "total_vcpus",
            "smt_ratio",
            "smt_util_overhead",
            "nosmt_util_overhead",
            "embodied_carbon_kg",
            "server_cost_usd",
            "carbon_intensity_g_kwh",
            "electricity_cost_usd_kwh",
            "lifetime_years",
            "nosmt_physical_cores",
            "nosmt_power_ratio",
            "nosmt_idle_ratio",
            "smt_physical_cores",
            "smt_p_idle",
            "smt_p_max",
        ]


def load_config(path: str | Path) -> ExperimentConfig:
    """
    Load an experiment configuration from a JSON file.

    Supports JSON with comments (JSONC) if json5 is installed.

    Args:
        path: Path to JSON config file

    Returns:
        ExperimentConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If JSON is invalid
        ValueError: If config structure is invalid
    """
    path = Path(path)
    with open(path, 'r') as f:
        if _HAS_JSON5:
            data = json5.load(f)
        else:
            data = json.load(f)
    return ExperimentConfig.from_dict(data)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """
    Save an experiment configuration to a JSON file.

    Args:
        config: ExperimentConfig to save
        path: Path to output file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)


def validate_config(config: ExperimentConfig) -> List[str]:
    """
    Validate a configuration and return list of error messages.

    Returns empty list if config is valid.
    """
    errors = []

    # Check name
    if not config.name or not config.name.strip():
        errors.append("Config must have a non-empty 'name'")

    # Check utilization bounds
    if not 0.0 <= config.workload.avg_util <= 1.0:
        errors.append(f"avg_util must be in [0, 1], got {config.workload.avg_util}")

    # Check positive values
    if config.workload.total_vcpus <= 0:
        errors.append(f"total_vcpus must be positive, got {config.workload.total_vcpus}")

    # Check all processors have valid values
    for name, proc in config.processor.processors.items():
        if proc.physical_cores <= 0:
            errors.append(f"Processor '{name}' physical_cores must be positive")
        if proc.threads_per_core <= 0:
            errors.append(f"Processor '{name}' threads_per_core must be positive")
        if proc.power_idle_w < 0:
            errors.append(f"Processor '{name}' power_idle_w must be non-negative")
        if proc.power_max_w <= 0:
            errors.append(f"Processor '{name}' power_max_w must be positive")
        if proc.power_idle_w > proc.power_max_w:
            errors.append(f"Processor '{name}' power_idle_w must be <= power_max_w")

    if config.oversubscription.smt_ratio < 1.0:
        errors.append(f"smt_ratio must be >= 1.0, got {config.oversubscription.smt_ratio}")

    # Check power curve
    if config.power_curve.type == "power" and config.power_curve.exponent is None:
        errors.append("power_curve type 'power' requires 'exponent' parameter")

    if config.power_curve.type not in ("linear", "power", "specpower", "polynomial"):
        errors.append(f"Unknown power_curve type: {config.power_curve.type}")

    # Check sweep if present
    if config.sweep:
        valid_params = config.get_sweepable_parameters()
        if config.sweep.parameter not in valid_params:
            errors.append(f"Invalid sweep parameter: {config.sweep.parameter}. "
                         f"Valid: {valid_params}")
        if not config.sweep.values:
            errors.append("Sweep must have at least one value")

    return errors
