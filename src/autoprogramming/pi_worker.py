"""Implementation-only Pi workers and their isolated task bundles."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from .errors import RunnerError
from .pi_rpc import PiResult, PiUsage, assistant_failure
from .portfolio import AvenueSpec
from .resources import Resources


WORKER_SYSTEM = """You are the sole implementation engineer for a standalone
Python function. Your first and non-negotiable obligation is MECHANISM FIDELITY:
implement exactly the approach contract in task.md and push that particular
mechanism as far as possible. Returning a plausible answer through a different
approach is a failure, even if it avoids an exception or appears more reliable.

NEVER replace the assigned mechanism because a package, model, API credential,
GPU, network service, or other capability is missing in your authoring shell.
Third-party packages need not already be installed: declare them in the PEP 723
block so the execution controller can resolve them. If an assigned capability
really cannot be exercised locally, still implement it faithfully, syntax-check
what you can, and make the runtime fail clearly with a precise setup error. Do
not add classical ML, classical CV, rules, regex, lookup, local-model, or API
fallbacks from another approach family. Retries, parsing, preprocessing, and
error handling are welcome only when the assigned mechanism remains the path
that produces the answer. Cross-family routing is allowed solely when task.md
explicitly identifies this as a composition contract.

You have no broader coordination duties. Work only in the provided directory.
Read task.md and examples.jsonl, then create solution.py defining predict with
exactly the input parameters in task.md. solution.py may contain one valid PEP
723 script block for dependencies. Do not hard-code or copy example outputs into
a lookup table. Load clients and models lazily. If files are needed at runtime,
put them under the artifact namespace named in task.md and declare that same
directory as `artifact_namespace` under `[tool.ap]`; resolve it as
`Path(__file__).parents[1] / "artifacts" / <namespace>`. If a call spends money,
report AP_COST_DOLLARS after each prediction. Finish only after checking that no
error branch substitutes a different mechanism and syntax-checking solution.py.
"""


class PiWorkerRunner:
    """Run one implementation-only Pi worker in an isolated context."""

    def __init__(self, command: tuple[str, ...] = ("pi",), timeout: float = 1200.0):
        self.command = tuple(command)
        self.timeout = timeout

    def run(
        self,
        cwd: Path,
        task: str,
        *,
        model: str | None = None,
        session_id: str | None = None,
        allowed_api_providers: tuple[str, ...] = (),
    ) -> PiResult:
        guard_source = Path(__file__).parent / "pi" / "worker-guard.ts"
        guard = cwd / ".tools" / "root-guard.ts"
        guard.parent.mkdir(exist_ok=True)
        shutil.copyfile(guard_source, guard)
        args = [
            *self.command,
            "--mode", "json", "--print",
            "--no-extensions", "--extension", str(guard),
            "--no-skills", "--no-prompt-templates",
            "--no-themes", "--no-context-files", "--no-approve",
            "--tools", "read,bash,edit,write",
            "--system-prompt", WORKER_SYSTEM,
        ]
        if session_id:
            session_dir = cwd / ".pi-sessions"
            session_dir.mkdir(exist_ok=True)
            args.extend(("--session-dir", str(session_dir), "--session-id", session_id))
        else:
            args.append("--no-session")
        if model:
            args.extend(("--model", model))
        args.append(task)
        try:
            proc = subprocess.run(
                args,
                cwd=str(cwd),
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
                env=worker_env(
                    allowed_api_providers,
                    pi_model=model,
                ),
            )
        except FileNotFoundError as exc:
            raise RunnerError(
                f"Pi worker executable {self.command[0]!r} was not found."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RunnerError(
                f"Pi implementation worker timed out after {self.timeout:g}s."
            ) from exc

        messages: list[dict] = []
        usage = PiUsage()
        text = ""
        for line in proc.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "message_end" and isinstance(
                event.get("message"), dict
            ):
                msg = event["message"]
                messages.append(msg)
                usage.add_message(msg)
                if msg.get("role") == "assistant":
                    parts = [
                        part.get("text", "")
                        for part in msg.get("content", [])
                        if part.get("type") == "text"
                    ]
                    if parts:
                        text = "".join(parts)
        if proc.returncode != 0:
            raise RunnerError(
                f"Pi implementation worker exited with code {proc.returncode}: "
                f"{proc.stderr[-2000:]}"
            )
        failure = assistant_failure(messages)
        if failure:
            raise RunnerError(f"{failure}\n{proc.stderr[-2000:]}")
        return PiResult(
            text=text,
            usage=usage,
            messages=messages,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )


_PROVIDER_ENV_MARKERS = {
    "anthropic": ("ANTHROPIC_", "ANT_LING_"),
    "openai": ("OPENAI_", "AZURE_OPENAI_"),
    "google": ("GEMINI_", "GOOGLE_"),
    "groq": ("GROQ_",),
    "mistral": ("MISTRAL_",),
    "openrouter": ("OPENROUTER_",),
    "together": ("TOGETHER_",),
    "fireworks": ("FIREWORKS_",),
    "deepseek": ("DEEPSEEK_",),
    "xai": ("XAI_",),
    "aws": ("AWS_",),
}


def worker_env(
    allowed_api_providers: tuple[str, ...] = (), *, pi_model: str | None = None
) -> dict[str, str]:
    """Scrub controller state and candidate API credentials outside allowlists."""
    allowed = {str(name).lower() for name in allowed_api_providers}
    if pi_model:
        lower_model = pi_model.lower()
        if "/" in lower_model:
            allowed.add(lower_model.split("/", 1)[0])
        elif any(
            token in lower_model for token in ("claude", "sonnet", "haiku", "opus")
        ):
            allowed.add("anthropic")
        elif "gemini" in lower_model:
            allowed.add("google")
        elif any(token in lower_model for token in ("gpt", "o3", "o4")):
            allowed.add("openai")
    allowed_markers = tuple(
        marker
        for provider in allowed
        for marker in _PROVIDER_ENV_MARKERS.get(
            provider, (provider.upper() + "_",)
        )
    )
    all_markers = tuple(
        marker for markers in _PROVIDER_ENV_MARKERS.values() for marker in markers
    )
    result: dict[str, str] = {}
    for key, value in os.environ.items():
        if (
            key == "AP_WORKSPACE"
            or key.startswith("AUTOPROGRAMMING_")
            or key == "PYTHONPATH"
        ):
            continue
        credential_like = (
            key.endswith("_API_KEY")
            or key.endswith("_OAUTH_TOKEN")
            or key
            in (
                "AWS_SECRET_ACCESS_KEY",
                "AWS_ACCESS_KEY_ID",
                "AWS_BEARER_TOKEN_BEDROCK",
            )
            or key.startswith(all_markers)
        )
        if credential_like and not key.startswith(allowed_markers):
            continue
        result[key] = value
    return result


def materialize_bundle(
    source: str, sandbox: Path, workspace, namespace: str
) -> str:
    """Copy worker artifacts into an immutable per-candidate namespace."""
    from .candidates import next_name

    candidate_name = next_name(workspace)
    versioned = f"{namespace}-{candidate_name}"
    source = source.replace(f'"{namespace}"', f'"{versioned}"')
    source = source.replace(f"'{namespace}'", f"'{versioned}'")
    source_artifacts = sandbox / "artifacts" / namespace
    if source_artifacts.is_dir():
        from .candidates import parse_pep723

        metadata = parse_pep723(source) or {}
        declared = ((metadata.get("tool") or {}).get("ap") or {}).get(
            "artifact_namespace"
        )
        if declared != versioned:
            raise RunnerError(
                "Worker created runtime artifacts but solution.py does not declare "
                f"[tool.ap] artifact_namespace = {versioned!r}."
            )
        target = Path(workspace.artifacts_dir) / versioned
        if target.exists():
            candidate_path = Path(workspace.candidates_dir) / f"{candidate_name}.py"
            if candidate_path.exists():
                raise RunnerError(
                    f"Refusing to overwrite candidate artifact bundle {target}."
                )
            # The controller journals the expected candidate name before bundle
            # import. A crash after copy but before candidate creation can leave
            # only this namespace; it is safe to replace because no candidate
            # file can reference it yet.
            shutil.rmtree(target)
        shutil.copytree(source_artifacts, target)
    return source


def task_document(schema, spec: AvenueSpec, resources: Resources) -> str:
    inputs = "\n".join(
        f"- `{field.name}: {field.type_name}` — {field.description}"
        for field in schema.inputs
    )
    outputs = "\n".join(
        f"- `{field.name}: {field.type_name}` — {field.description}"
        for field in schema.outputs
    )
    search_resources = {
        key: getattr(resources.search, key)
        for key in (
            "cpu_cores", "memory_gb", "disk_gb", "gpu", "gpu_vram_gb",
            "allow_package_installs", "allow_model_downloads", "fine_tuning",
        )
    }
    search_resources["available_runtime_api_access"] = (
        resources.search.candidate_api_providers
    )
    return f"""# Implementation task

## Goal
{schema.doc}

## Inputs
{inputs}

## Outputs
{outputs}

`predict` must accept exactly: {', '.join(schema.input_names)}.
For one output, return its value. For several outputs, return a tuple in the
order above or a dict keyed by output name.

## Non-negotiable approach contract
{spec.title}: {spec.implementation_brief}

Hypothesis: {spec.hypothesis}
Required mechanism boundary: {spec.mechanism}
Required mechanism evidence: {list(spec.required_mechanisms)}
Forbidden substitutions: {list(spec.forbidden_substitutions)}
Cross-tier fallback permitted: {spec.allow_cross_tier_fallback}

The implementation is invalid if another family produces the answer when this
mechanism is unavailable. Missing dependencies or capabilities must cause a
clear setup/runtime failure, never a substitute implementation.

## Available build/search resources
{json.dumps(search_resources, default=str, indent=2)}

Packages declared in solution.py's PEP 723 block are resolved by the execution
controller. Their absence from the current shell is not a reason to avoid them.

## Permitted runtime resources
{json.dumps(resources.runtime.__dict__, default=str, indent=2)}
Allowed API providers for this implementation: {list(spec.allowed_api_providers)}

## Runtime artifact namespace
If runtime files are needed, use `artifacts/{spec.id}/` here and declare
`artifact_namespace = "{spec.id}"` under `[tool.ap]` in solution.py. At runtime
that directory is `Path(__file__).parents[1] / "artifacts" / "{spec.id}"`.
"""


def worker_run_dir(workspace) -> Path:
    """Opaque worker root outside the optimizer/package workspace."""
    try:
        token = workspace.active.get("private_data_id")
    except Exception:
        token = None
    if not token:
        token = hashlib.sha256(
            str(Path(workspace.root).resolve()).encode()
        ).hexdigest()[:24]
    base = Path(os.environ.get("AP_WORKER_DIR", Path.home() / ".cache" / "ap-work"))
    path = base / str(token)
    path.mkdir(parents=True, exist_ok=True)
    return path


def avenue_dir(workspace, avenue_id: str) -> Path:
    return worker_run_dir(workspace) / avenue_id
