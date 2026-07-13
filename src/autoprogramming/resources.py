"""Resource and data-governance contracts for an optimization run.

AutoProgramming must not infer that a build machine's capabilities are also
available to the shipped program.  The objects in this module keep search-time
resources, deployment-time resources, and data-egress policy separate.  They
contain capability names and limits only -- never API keys or other secrets.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .errors import AutoProgrammingError


class ResourceError(AutoProgrammingError):
    """A resource profile is invalid or has not been confirmed."""


def _positive(name: str, value, *, allow_none: bool = True) -> None:
    if value is None and allow_none:
        return
    if value is None or not math.isfinite(value) or value <= 0:
        raise ResourceError(f"{name} must be positive and finite, got {value!r}.")


def _tuple(value) -> tuple:
    if value is None:
        return ()
    return tuple(value)


@dataclass(frozen=True)
class DataPolicy:
    """Where task data may go during search and at runtime.

    ``external_egress`` is deliberately tri-state.  ``None`` means the user has
    not answered yet; a networked approach must never treat that as consent.
    """

    external_egress: bool | None = None
    allowed_domains: tuple[str, ...] = ()
    sensitive: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_domains", _tuple(self.allowed_domains))


@dataclass(frozen=True)
class SearchResources:
    """Resources available while candidates are researched and built."""

    cpu_cores: int | None = None
    memory_gb: float | None = None
    disk_gb: float | None = None
    gpu: str | None = None
    gpu_vram_gb: float | None = None
    max_parallel_agents: int = 4
    max_dollars_per_agent_call: float | None = None
    allow_package_installs: bool | None = None
    allow_model_downloads: bool | None = None
    fine_tuning: bool = False
    pi_models: tuple[str, ...] = ()
    pi_local: bool = False
    candidate_api_providers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _positive("search.cpu_cores", self.cpu_cores)
        _positive("search.memory_gb", self.memory_gb)
        _positive("search.disk_gb", self.disk_gb)
        _positive("search.gpu_vram_gb", self.gpu_vram_gb)
        _positive("search.max_parallel_agents", self.max_parallel_agents, allow_none=False)
        _positive(
            "search.max_dollars_per_agent_call",
            self.max_dollars_per_agent_call,
        )
        object.__setattr__(self, "pi_models", _tuple(self.pi_models))
        object.__setattr__(
            self, "candidate_api_providers", _tuple(self.candidate_api_providers)
        )

    @classmethod
    def detect(cls) -> "SearchResources":
        """Conservatively detect local hardware; never infer network consent."""
        cores = os.cpu_count() or 1
        memory_gb = None
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            memory_gb = pages * page_size / (1024 ** 3)
        except (AttributeError, OSError, ValueError):
            pass
        disk_gb = shutil.disk_usage(Path.cwd()).free / (1024 ** 3)

        gpu = None
        vram = None
        if shutil.which("nvidia-smi"):
            try:
                proc = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                )
                line = proc.stdout.splitlines()[0] if proc.stdout.splitlines() else ""
                if line:
                    name, _, memory = line.rpartition(",")
                    gpu = name.strip() or "cuda"
                    vram = float(memory.strip()) / 1024 if memory.strip() else None
            except (OSError, ValueError, subprocess.SubprocessError):
                gpu = "cuda"

        return cls(
            cpu_cores=cores,
            memory_gb=memory_gb,
            disk_gb=disk_gb,
            gpu=gpu,
            gpu_vram_gb=vram,
            max_parallel_agents=max(1, min(4, cores)),
            # These require consent and deliberately remain unanswered.
            allow_package_installs=None,
            allow_model_downloads=None,
        )


@dataclass(frozen=True)
class RuntimeResources:
    """The resource envelope of the package that will actually be shipped."""

    network: bool | None = None
    api_providers: tuple[str, ...] = ()
    agent_runtime: bool = False
    gpu: str | None = None
    memory_gb: float | None = None
    disk_gb: float | None = None
    max_dollars_per_call: float | None = None
    max_latency_ms: float | None = None
    max_artifact_mb: float | None = None
    offline: bool = False

    def __post_init__(self) -> None:
        _positive("runtime.memory_gb", self.memory_gb)
        _positive("runtime.disk_gb", self.disk_gb)
        _positive("runtime.max_dollars_per_call", self.max_dollars_per_call)
        _positive("runtime.max_latency_ms", self.max_latency_ms)
        _positive("runtime.max_artifact_mb", self.max_artifact_mb)
        object.__setattr__(self, "api_providers", _tuple(self.api_providers))
        if self.offline and self.network is True:
            raise ResourceError("runtime.offline=True conflicts with runtime.network=True.")
        if self.offline and self.api_providers:
            raise ResourceError(
                "An offline runtime cannot declare external API providers."
            )


@dataclass(frozen=True)
class Resources:
    """Confirmed capabilities and constraints for one optimization run."""

    search: SearchResources = field(default_factory=SearchResources.detect)
    runtime: RuntimeResources = field(default_factory=RuntimeResources)
    data: DataPolicy = field(default_factory=DataPolicy)
    confirmed: bool = False

    @classmethod
    def detect(cls) -> "Resources":
        """Detect hardware and leave consent-sensitive fields unanswered."""
        return cls(search=SearchResources.detect())

    @property
    def questions(self) -> tuple[str, ...]:
        """Questions still requiring a user answer before agentic search."""
        questions: list[str] = []
        if self.data.external_egress is None:
            questions.append("May task examples or derived content leave this machine?")
        if self.runtime.network is None:
            questions.append("May the shipped program use the network at runtime?")
        if self.search.allow_package_installs is None:
            questions.append("May workers install third-party Python packages?")
        if self.search.allow_model_downloads is None:
            questions.append("May workers download pretrained model artifacts?")
        if not self.confirmed:
            questions.append("Confirm this search and deployment resource profile.")
        return tuple(questions)

    def ensure_confirmed(self) -> None:
        if self.questions:
            rendered = "\n".join(f"  - {q}" for q in self.questions)
            raise ResourceError(
                "Resource profile is incomplete. AutoProgramming will not guess "
                "about data egress, package installation, model downloads, or "
                f"deployment networking. Answer and confirm:\n{rendered}"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "Resources":
        return cls(
            search=SearchResources(**value.get("search", {})),
            runtime=RuntimeResources(**value.get("runtime", {})),
            data=DataPolicy(**value.get("data", {})),
            confirmed=bool(value.get("confirmed", False)),
        )

    @property
    def pi_may_receive_task_data(self) -> bool:
        """Whether Pi workers may legally inspect examples and task context."""
        return self.search.pi_local or self.data.external_egress is True

    def feasibility(self) -> dict[int, dict[str, str | bool]]:
        """Feasibility of the approach ladder under this runtime contract."""
        runtime_network = self.runtime.network is True and not self.runtime.offline
        APIs = bool(self.runtime.api_providers)
        can_externalize = self.data.external_egress is True
        install = self.search.allow_package_installs is True
        download = self.search.allow_model_downloads is True

        def item(ok: bool, reason: str) -> dict[str, str | bool]:
            return {"feasible": ok, "reason": reason}

        result = {
            1: item(
                runtime_network and self.runtime.agent_runtime and can_externalize,
                "requires a runtime coding/generalist agent, network, and permitted data egress",
            ),
            2: item(
                runtime_network and APIs and can_externalize,
                "requires runtime model APIs and permitted data egress",
            ),
            3: item(
                runtime_network and APIs and can_externalize,
                "requires a runtime model API and permitted data egress",
            ),
            4: item(
                self.search.fine_tuning
                and (
                    (APIs and can_externalize)
                    or self.runtime.gpu is not None
                    or (self.runtime.disk_gb or 0) > 0
                ),
                "requires fine-tuning access plus a deployable endpoint or local runtime",
            ),
            5: item(
                install and download and (
                    self.runtime.gpu is not None
                    or (self.runtime.memory_gb or 0) >= 2
                    or (self.runtime.disk_gb or 0) >= 2
                ),
                "requires package/model downloads and sufficient deployment compute",
            ),
            6: item(
                install,
                "requires permission to install classical-ML dependencies",
            ),
            7: item(True, "stdlib algorithms, rules, and feature engineering are always feasible"),
        }
        result[8] = item(
            sum(bool(v["feasible"]) for k, v in result.items() if k <= 7) >= 2,
            "composition requires at least two feasible implementation families",
        )
        return result
