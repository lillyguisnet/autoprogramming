"""Execute candidates in isolated subprocess environments.

``run_candidate`` is the one-shot trace/production primitive. ``CandidateSession``
keeps one process alive across an evaluation split so lazy clients and models
persist and warm latency is not confused with cold start. Drivers live under
the workspace's ``.ap/`` directory and use plain ``sys.executable`` when the
candidate has no third-party dependencies (fast path, no uv needed),
``uv run --no-project`` otherwise. The driver reports back through a
per-run result file, never through stdout — a candidate is free to print
anything (including text that looks like a result report) without
corrupting or forging the run's outcome. A candidate raising inside
``predict()`` comes back as a failed RunResult — never a RunnerError,
which is reserved for the harness itself failing (e.g. uv missing).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import textwrap
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .candidates import Candidate, pep503_normalize, runtime_deps
from .errors import RunnerError

DEFAULT_TIMEOUT = 120.0
STDERR_TAIL = 2000

_BARE_TOML_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")

_DRIVER_BODY = '''
import importlib.util
import json
import math
import sys
import traceback

sys.path.insert(0, _PARENT_DIR)

_BASES = {
    "bool": bool, "int": int, "float": float, "complex": complex,
    "str": str, "bytes": bytes, "list": list, "tuple": tuple,
    "dict": dict, "set": set, "frozenset": frozenset,
}


def _load_candidate():
    spec = importlib.util.spec_from_file_location("_ap_candidate", _CANDIDATE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_ap_candidate"] = module
    spec.loader.exec_module(module)
    return module


def _map_outputs(value):
    names = [f["name"] for f in _OUTPUT_SPEC]
    if isinstance(value, dict):
        missing = [n for n in names if n not in value]
        if missing:
            raise ValueError("predict() returned a dict missing outputs: " + repr(missing))
        mapped = {n: value[n] for n in names}
    elif len(names) == 1:
        mapped = {names[0]: value}
    elif isinstance(value, (tuple, list)) and len(value) == len(names):
        mapped = dict(zip(names, value))
    else:
        raise ValueError(
            "predict() must return " + str(len(names)) + " outputs ("
            + ", ".join(names) + ") as a tuple in schema order, got " + repr(value)
        )
    wire = {}
    for f in _OUTPUT_SPEC:
        base = _BASES[f["base"]]
        v = mapped[f["name"]]
        wire[f["name"]] = v if type(v) is base else base(v)
    return wire


def _main():
    with open(sys.argv[1], encoding="utf-8") as fh:
        inputs = json.load(fh)
    try:
        module = _load_candidate()
        outputs = _map_outputs(module.predict(**inputs))
        cost = getattr(module, "AP_COST_DOLLARS", None)
        if cost is not None:
            try:
                cost = float(cost)
                if not math.isfinite(cost) or cost < 0:
                    cost = None
            except (TypeError, ValueError):
                cost = None
        payload = {"ok": True, "outputs": outputs, "cost_dollars": cost}
    except Exception:
        payload = {"ok": False, "error": traceback.format_exc()}
    with open(_RESULT_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


_main()
'''

_SESSION_DRIVER_BODY = '''
import importlib.util
import json
import math
import os
import sys
import time
import traceback

sys.path.insert(0, _PARENT_DIR)
_BASES = {
    "bool": bool, "int": int, "float": float, "complex": complex,
    "str": str, "bytes": bytes, "list": list, "tuple": tuple,
    "dict": dict, "set": set, "frozenset": frozenset,
}


def _load_candidate():
    spec = importlib.util.spec_from_file_location("_ap_candidate", _CANDIDATE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_ap_candidate"] = module
    spec.loader.exec_module(module)
    return module


def _map_outputs(value):
    names = [f["name"] for f in _OUTPUT_SPEC]
    if isinstance(value, dict):
        missing = [n for n in names if n not in value]
        if missing:
            raise ValueError("predict() returned a dict missing outputs: " + repr(missing))
        mapped = {n: value[n] for n in names}
    elif len(names) == 1:
        mapped = {names[0]: value}
    elif isinstance(value, (tuple, list)) and len(value) == len(names):
        mapped = dict(zip(names, value))
    else:
        raise ValueError("predict() returned the wrong output shape: " + repr(value))
    wire = {}
    for f in _OUTPUT_SPEC:
        base = _BASES[f["base"]]
        value = mapped[f["name"]]
        wire[f["name"]] = value if type(value) is base else base(value)
    return wire


try:
    _MODULE = _load_candidate()
    _IMPORT_ERROR = None
except Exception:
    _MODULE = None
    _IMPORT_ERROR = traceback.format_exc()

for _line in sys.stdin:
    try:
        request = json.loads(_line)
    except Exception:
        continue
    if request.get("stop"):
        break
    result_path = request["result"]
    started = time.monotonic()
    if _IMPORT_ERROR is not None:
        payload = {"ok": False, "error": _IMPORT_ERROR}
    else:
        try:
            with open(request["input"], encoding="utf-8") as fh:
                inputs = json.load(fh)
            outputs = _map_outputs(_MODULE.predict(**inputs))
            cost = getattr(_MODULE, "AP_COST_DOLLARS", None)
            if cost is not None:
                try:
                    cost = float(cost)
                    if not math.isfinite(cost) or cost < 0:
                        cost = None
                except (TypeError, ValueError):
                    cost = None
            payload = {"ok": True, "outputs": outputs, "cost_dollars": cost}
        except Exception:
            payload = {"ok": False, "error": traceback.format_exc()}
    payload["predict_duration_s"] = time.monotonic() - started
    temp_result_path = result_path + ".tmp"
    with open(temp_result_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    os.replace(temp_result_path, result_path)
'''


@dataclass
class RunResult:
    """The outcome of running one candidate on one row of inputs."""

    ok: bool
    outputs: dict | None
    error: str | None
    stdout: str
    stderr: str
    duration_s: float
    cost_dollars: float | None
    candidate: str
    inputs: dict
    cold_start: bool = False

    def trace(self) -> str:
        """A readable block: candidate, inputs, outputs or error, stdio, timing."""
        status = "ok" if self.ok else "FAILED"
        lines = [f"candidate {self.candidate} ({status}, {self.duration_s:.3f}s)"]
        lines.append(f"inputs:  {json.dumps(self.inputs, default=repr)}")
        if self.ok:
            lines.append(f"outputs: {json.dumps(self.outputs, default=repr)}")
        if self.error:
            lines.append("error:")
            lines.append(textwrap.indent(self.error.rstrip("\n"), "  "))
        if self.stdout.strip():
            lines.append("stdout:")
            lines.append(textwrap.indent(self.stdout.rstrip("\n"), "  "))
        if self.stderr.strip():
            lines.append("stderr:")
            lines.append(textwrap.indent(self.stderr.rstrip("\n"), "  "))
        if self.cost_dollars is not None:
            lines.append(f"cost: ${self.cost_dollars:g}")
        if self.cold_start:
            lines.append("latency phase: cold start")
        return "\n".join(lines)


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        inner = ", ".join(f"{_toml_key(k)} = {_toml_value(v)}" for k, v in value.items())
        return "{ " + inner + " }"
    raise RunnerError(
        f"Cannot render {value!r} into the driver's PEP 723 block — the "
        f"candidate metadata must stay expressible as TOML so uv can read it. "
        f"Simplify the offending entry in the candidate's `# /// script` block."
    )


def _toml_key(key: str) -> str:
    return key if _BARE_TOML_KEY_RE.match(key) else json.dumps(key)


def _driver_pep723(candidate: Candidate, deps: tuple[str, ...], dist_name: str) -> str:
    lines = ["# /// script"]
    if candidate.requires_python:
        lines.append(f"# requires-python = {_toml_value(candidate.requires_python)}")
    lines.append("# dependencies = [")
    for dep in deps:
        lines.append(f"#     {_toml_value(dep)},")
    lines.append("# ]")
    self_norm = pep503_normalize(dist_name)
    sources = {
        key: value
        for key, value in candidate.uv_sources.items()
        if pep503_normalize(key) != self_norm
    }
    if sources:
        lines.append("#")
        lines.append("# [tool.uv.sources]")
        for key, value in sources.items():
            lines.append(f"# {_toml_key(key)} = {_toml_value(value)}")
    lines.append("# ///")
    return "\n".join(lines) + "\n"


def _driver_source(
    candidate: Candidate,
    deps: tuple[str, ...],
    dist_name: str,
    parent_dir: str,
    output_spec: list[dict],
    result_path: str | Path,
) -> str:
    block = _driver_pep723(candidate, deps, dist_name) if deps else ""
    prelude = (
        f"_CANDIDATE_PATH = {str(Path(candidate.path).resolve())!r}\n"
        f"_PARENT_DIR = {parent_dir!r}\n"
        f"_OUTPUT_SPEC = {output_spec!r}\n"
        f"_RESULT_PATH = {str(result_path)!r}\n"
    )
    return block + prelude + _DRIVER_BODY


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _kill_process_group(proc: subprocess.Popen) -> None:
    """SIGKILL the run's whole process group, then the direct child.

    A timed-out candidate must actually stop spending: ``uv run`` executes
    the driver as a grandchild, and predict() may spawn subprocesses of its
    own, so killing only the direct child would leave the real work running
    (and billing APIs) as an orphan. The child was started in its own
    session, so its process-group id is its pid and the group kill takes the
    whole tree down.
    """
    if hasattr(os, "killpg"):
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(OSError):
        proc.kill()


def run_candidate(
    workspace, candidate: Candidate, inputs: dict, timeout: float = DEFAULT_TIMEOUT
) -> RunResult:
    """Run ``candidate.predict(**inputs)`` in a subprocess and collect a RunResult.

    Stdlib-only candidates run under ``sys.executable`` directly; candidates
    with third-party dependencies run under ``uv run --no-project`` with a
    driver carrying their PEP 723 block minus the self-reference (the
    workspace package is injected via sys.path instead of an editable
    install, so the run is hermetic). The driver reports through a per-run
    result file, so candidate stdout — whatever it contains — never decides
    the outcome. predict() exceptions and timeouts come back as ``ok=False``
    results (a timeout kills the run's entire process group, uv grandchild
    and candidate-spawned subprocesses included); only harness failures
    raise RunnerError.
    """
    tmp_dir = Path(workspace.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    root = Path(workspace.root).resolve()
    parent_dir = str(root.parent)
    output_spec = [
        {"name": f.name, "base": f.base.__name__} for f in workspace.schema.outputs
    ]

    deps = runtime_deps(candidate, workspace.dist_name)
    if deps and shutil.which("uv") is None:
        raise RunnerError(
            f"Cannot run {candidate.name}: it declares third-party dependencies "
            f"({', '.join(deps)}), which are executed in an ephemeral uv-resolved "
            f"environment so candidates with conflicting dependencies can coexist "
            f"— but `uv` was not found on PATH. Install uv "
            f"(https://docs.astral.sh/uv/) or make the candidate stdlib-only to "
            f"use the no-deps fast path."
        )

    token = uuid.uuid4().hex
    driver_path = tmp_dir / f"driver_{token}.py"
    inputs_path = tmp_dir / f"inputs_{token}.json"
    result_path = tmp_dir / f"result_{token}.json"

    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = parent_dir + (os.pathsep + existing if existing else "")
    env["AP_WORKSPACE"] = str(root)

    if deps:
        cmd = ["uv", "run", "--no-project", "--quiet", str(driver_path), str(inputs_path)]
    else:
        cmd = [sys.executable, str(driver_path), str(inputs_path)]

    start = time.monotonic()
    payload_text: str | None = None
    try:
        driver_path.write_text(
            _driver_source(
                candidate, deps, workspace.dist_name, parent_dir, output_spec,
                result_path,
            ),
            encoding="utf-8",
        )
        inputs_path.write_text(json.dumps(inputs), encoding="utf-8")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=str(root),
                start_new_session=(os.name == "posix"),
            )
        except FileNotFoundError as exc:
            raise RunnerError(
                f"Could not launch the run process for {candidate.name} "
                f"({cmd[0]!r}: {exc}). Candidates execute in their own "
                f"subprocess, so the interpreter (or uv) must be runnable; "
                f"check your PATH and try again."
            ) from exc
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_process_group(proc)
            stdout, stderr = proc.communicate()
            return RunResult(
                ok=False,
                outputs=None,
                error=f"timed out after {timeout}s",
                stdout=_as_text(stdout),
                stderr=_as_text(stderr),
                duration_s=time.monotonic() - start,
                cost_dollars=candidate.cost_per_call,
                candidate=candidate.name,
                inputs=dict(inputs),
            )
        if result_path.exists():
            payload_text = result_path.read_text(encoding="utf-8")
    finally:
        for path in (driver_path, inputs_path, result_path):
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)

    duration = time.monotonic() - start
    stdout = stdout or ""
    stderr = stderr or ""
    payload = None
    payload_error = None
    if payload_text is not None:
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            payload_error = (
                f"malformed result payload from the driver ({exc}): "
                f"{payload_text[:200]!r}"
            )

    if payload is None:
        parts = [
            part
            for part in (payload_error, stderr[-STDERR_TAIL:] if stderr.strip() else None)
            if part
        ]
        error = "\n".join(parts) or (
            f"candidate process exited with code {proc.returncode} without "
            f"reporting a result or printing any stderr"
        )
        return RunResult(
            ok=False,
            outputs=None,
            error=error,
            stdout=stdout,
            stderr=stderr,
            duration_s=duration,
            cost_dollars=candidate.cost_per_call,
            candidate=candidate.name,
            inputs=dict(inputs),
        )

    cost = payload.get("cost_dollars")
    if cost is None:
        cost = candidate.cost_per_call
    if payload.get("ok"):
        return RunResult(
            ok=True,
            outputs=payload.get("outputs"),
            error=None,
            stdout=stdout,
            stderr=stderr,
            duration_s=duration,
            cost_dollars=cost,
            candidate=candidate.name,
            inputs=dict(inputs),
        )
    return RunResult(
        ok=False,
        outputs=None,
        error=str(payload.get("error") or "predict() failed with no traceback"),
        stdout=stdout,
        stderr=stderr,
        duration_s=duration,
        cost_dollars=cost,
        candidate=candidate.name,
        inputs=dict(inputs),
    )


class CandidateSession:
    """A candidate process reused across rows so lazy globals actually persist.

    Requests and results travel through throwaway files; candidate stdout is
    never protocol data. The first call is marked cold and includes candidate
    import/model initialization. Later calls measure warm repeated inference.
    """

    def __init__(
        self, workspace, candidate: Candidate, timeout: float = DEFAULT_TIMEOUT
    ):
        self.workspace = workspace
        self.candidate = candidate
        self.timeout = timeout
        self._proc: subprocess.Popen | None = None
        self._driver_path: Path | None = None
        self._tmp_paths: set[Path] = set()
        self._stdout: list[str] = []
        self._stderr: list[str] = []
        self._lock = threading.Lock()
        self._calls = 0
        self._process_started_at: float | None = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_exc):
        self.close()

    def start(self) -> None:
        if self._proc is not None:
            return
        tmp_dir = Path(self.workspace.tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        root = Path(self.workspace.root).resolve()
        parent_dir = str(root.parent)
        output_spec = [
            {"name": f.name, "base": f.base.__name__}
            for f in self.workspace.schema.outputs
        ]
        deps = runtime_deps(self.candidate, self.workspace.dist_name)
        if deps and shutil.which("uv") is None:
            raise RunnerError(
                f"Cannot run {self.candidate.name}: it declares third-party "
                "dependencies but uv was not found on PATH."
            )
        token = uuid.uuid4().hex
        self._driver_path = tmp_dir / f"session_driver_{token}.py"
        block = _driver_pep723(self.candidate, deps, self.workspace.dist_name) if deps else ""
        prelude = (
            f"_CANDIDATE_PATH = {str(Path(self.candidate.path).resolve())!r}\n"
            f"_PARENT_DIR = {parent_dir!r}\n"
            f"_OUTPUT_SPEC = {output_spec!r}\n"
        )
        self._driver_path.write_text(block + prelude + _SESSION_DRIVER_BODY, encoding="utf-8")
        cmd = (
            ["uv", "run", "--no-project", "--quiet", str(self._driver_path)]
            if deps
            else [sys.executable, str(self._driver_path)]
        )
        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = parent_dir + (os.pathsep + existing if existing else "")
        env["AP_WORKSPACE"] = str(root)
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self._process_started_at = time.monotonic()
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=str(root),
                start_new_session=(os.name == "posix"),
                bufsize=1,
            )
        except FileNotFoundError as exc:
            self._cleanup_files()
            raise RunnerError(f"Could not launch candidate session: {exc}") from exc
        threading.Thread(
            target=self._drain, args=(self._proc.stdout, self._stdout), daemon=True
        ).start()
        threading.Thread(
            target=self._drain, args=(self._proc.stderr, self._stderr), daemon=True
        ).start()

    @staticmethod
    def _drain(stream, target: list[str]) -> None:
        if stream is None:
            return
        for chunk in stream:
            target.append(chunk)

    def run(self, inputs: dict) -> RunResult:
        self.start()
        assert self._proc is not None and self._proc.stdin is not None
        with self._lock:
            token = uuid.uuid4().hex
            tmp_dir = Path(self.workspace.tmp_dir)
            input_path = tmp_dir / f"session_input_{token}.json"
            result_path = tmp_dir / f"session_result_{token}.json"
            temp_result_path = Path(str(result_path) + ".tmp")
            self._tmp_paths.update((input_path, result_path, temp_result_path))
            input_path.write_text(json.dumps(inputs), encoding="utf-8")
            stdout_at = len(self._stdout)
            stderr_at = len(self._stderr)
            cold = self._calls == 0
            self._calls += 1
            request_started = time.monotonic()
            started = (
                self._process_started_at
                if cold and self._process_started_at is not None
                else request_started
            )
            try:
                self._proc.stdin.write(json.dumps({
                    "input": str(input_path), "result": str(result_path)
                }) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                return self._dead_result(inputs, started, cold)

            deadline = started + self.timeout
            while not result_path.exists():
                if self._proc.poll() is not None:
                    return self._dead_result(inputs, started, cold)
                if time.monotonic() >= deadline:
                    _kill_process_group(self._proc)
                    return RunResult(
                        ok=False, outputs=None,
                        error=f"timed out after {self.timeout}s",
                        stdout="".join(self._stdout[stdout_at:]),
                        stderr="".join(self._stderr[stderr_at:]),
                        duration_s=time.monotonic() - started,
                        cost_dollars=self.candidate.cost_per_call,
                        candidate=self.candidate.name, inputs=dict(inputs),
                        cold_start=cold,
                    )
                time.sleep(0.005)
            elapsed = time.monotonic() - started
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                payload = {"ok": False, "error": f"malformed session result: {exc}"}
            finally:
                for path in (input_path, result_path, temp_result_path):
                    with contextlib.suppress(OSError):
                        path.unlink(missing_ok=True)
                    self._tmp_paths.discard(path)
            duration = (
                elapsed
                if cold
                else float(payload.get("predict_duration_s") or elapsed)
            )
            cost = payload.get("cost_dollars")
            if cost is None:
                cost = self.candidate.cost_per_call
            common = dict(
                stdout="".join(self._stdout[stdout_at:]),
                stderr="".join(self._stderr[stderr_at:]),
                duration_s=duration,
                cost_dollars=cost,
                candidate=self.candidate.name,
                inputs=dict(inputs),
                cold_start=cold,
            )
            if payload.get("ok"):
                return RunResult(ok=True, outputs=payload.get("outputs"), error=None, **common)
            return RunResult(
                ok=False, outputs=None,
                error=str(payload.get("error") or "predict() failed with no traceback"),
                **common,
            )

    def _dead_result(self, inputs: dict, started: float, cold: bool) -> RunResult:
        assert self._proc is not None
        error = "candidate session exited unexpectedly"
        if self._stderr:
            error += "\n" + "".join(self._stderr)[-STDERR_TAIL:]
        return RunResult(
            ok=False, outputs=None, error=error,
            stdout="".join(self._stdout), stderr="".join(self._stderr),
            duration_s=time.monotonic() - started,
            cost_dollars=self.candidate.cost_per_call,
            candidate=self.candidate.name, inputs=dict(inputs), cold_start=cold,
        )

    def close(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                assert proc.stdin is not None
                proc.stdin.write('{"stop": true}\n')
                proc.stdin.flush()
                proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                _kill_process_group(proc)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=2)
        self._cleanup_files()

    def _cleanup_files(self) -> None:
        paths = list(self._tmp_paths)
        if self._driver_path is not None:
            paths.append(self._driver_path)
        for path in paths:
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
        self._tmp_paths.clear()
