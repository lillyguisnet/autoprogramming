"""Resource contracts and approach feasibility."""

from __future__ import annotations

import json

import pytest

import autoprogramming as ap
from autoprogramming.resources import ResourceError


class ResourceLabel(str):
    pass


def confirmed_resources(**runtime_overrides):
    return ap.Resources(
        search=ap.SearchResources(
            cpu_cores=8,
            memory_gb=16,
            disk_gb=100,
            max_parallel_agents=4,
            allow_package_installs=True,
            allow_model_downloads=True,
            fine_tuning=True,
            candidate_api_providers=("openai",),
        ),
        runtime=ap.RuntimeResources(network=True, **runtime_overrides),
        data=ap.DataPolicy(external_egress=True),
        confirmed=True,
    )


def test_detect_is_conservative_about_consent():
    resources = ap.Resources.detect()
    assert resources.search.cpu_cores >= 1
    assert resources.search.memory_gb is None or resources.search.memory_gb > 0
    assert resources.data.external_egress is None
    assert resources.runtime.network is None
    assert resources.search.allow_package_installs is None
    assert resources.questions


def test_agent_call_reservation_bound_must_be_positive_and_finite():
    with pytest.raises(ResourceError, match="positive and finite"):
        ap.SearchResources(max_dollars_per_agent_call=0)
    with pytest.raises(ResourceError, match="positive and finite"):
        ap.SearchResources(max_dollars_per_agent_call=float("inf"))
    assert ap.SearchResources(
        max_dollars_per_agent_call=0.05
    ).max_dollars_per_agent_call == 0.05


def test_incomplete_profile_refused_with_questions():
    with pytest.raises(ResourceError, match="will not guess") as exc:
        ap.Resources.detect().ensure_confirmed()
    assert "leave this machine" in str(exc.value)
    assert "deployment" in str(exc.value)


def test_confirmed_profile_round_trips_json():
    original = confirmed_resources(
        api_providers=("openai",), agent_runtime=True, memory_gb=8
    )
    loaded = ap.Resources.from_dict(json.loads(json.dumps(original.to_dict())))
    assert loaded == original
    loaded.ensure_confirmed()


def test_offline_runtime_rejects_network_and_apis():
    with pytest.raises(ResourceError, match="conflicts"):
        ap.RuntimeResources(offline=True, network=True)
    with pytest.raises(ResourceError, match="offline"):
        ap.RuntimeResources(offline=True, api_providers=("openai",))


def test_feasibility_covers_full_ladder():
    resources = confirmed_resources(
        api_providers=("openai",), agent_runtime=True, gpu="cuda", memory_gb=8
    )
    feasibility = resources.feasibility()
    assert set(feasibility) == set(range(1, 9))
    assert all(item["feasible"] for item in feasibility.values())


def test_resource_confirmed_program_workspace_hides_val_and_test(tmp_path, monkeypatch):
    monkeypatch.setenv("AP_PRIVATE_DATA_DIR", str(tmp_path / "private"))
    class QuietBackend:
        def run(self, harness, context):
            return None

    @ap.program
    def classify(text: str) -> ResourceLabel:
        """Classify."""

    rows = [{"text": str(i), "ResourceLabel": str(i)} for i in range(10)]
    profile = ap.Resources(
        search=ap.SearchResources(
            max_parallel_agents=1,
            allow_package_installs=False,
            allow_model_downloads=False,
        ),
        runtime=ap.RuntimeResources(network=False),
        data=ap.DataPolicy(external_egress=False),
        confirmed=True,
    )
    classify.optimize(
        rows, ap.Budget(eval_calls=10), workspace=tmp_path / "classify_ap",
        resources=profile, backend=QuietBackend(),
    )
    assert classify.workspace.resources_json.exists()
    assert not (classify.workspace.data_dir / "val.csv").exists()
    assert not (classify.workspace.data_dir / "test.csv").exists()
    assert classify.workspace.split_path("val").exists()


def test_pi_data_access_requires_egress_or_confirmed_local_model():
    remote_private = ap.Resources(
        search=ap.SearchResources(
            pi_local=False,
            allow_package_installs=False,
            allow_model_downloads=False,
        ),
        runtime=ap.RuntimeResources(network=False),
        data=ap.DataPolicy(external_egress=False),
        confirmed=True,
    )
    assert remote_private.pi_may_receive_task_data is False
    assert ap.Resources(
        search=ap.SearchResources(
            pi_local=True,
            allow_package_installs=False,
            allow_model_downloads=False,
        ),
        runtime=ap.RuntimeResources(network=False),
        data=ap.DataPolicy(external_egress=False),
        confirmed=True,
    ).pi_may_receive_task_data is True


def test_runtime_api_needs_confirmed_candidate_evaluation_access():
    incomplete = ap.Resources(
        search=ap.SearchResources(
            allow_package_installs=True,
            allow_model_downloads=True,
            candidate_api_providers=None,
        ),
        runtime=ap.RuntimeResources(network=True, api_providers=("openai",)),
        data=ap.DataPolicy(external_egress=True),
        confirmed=True,
    )
    assert any("candidate evaluations" in question for question in incomplete.questions)
    assert incomplete.feasibility()[3]["feasible"] is False

    confirmed_absent = ap.Resources(
        search=ap.SearchResources(
            allow_package_installs=True,
            allow_model_downloads=True,
            candidate_api_providers=(),
        ),
        runtime=ap.RuntimeResources(network=True, api_providers=("openai",)),
        data=ap.DataPolicy(external_egress=True),
        confirmed=True,
    )
    confirmed_absent.ensure_confirmed()
    assert confirmed_absent.feasibility()[3]["feasible"] is False


def test_offline_minimal_profile_keeps_code_and_rules():
    resources = ap.Resources(
        search=ap.SearchResources(
            max_parallel_agents=1,
            allow_package_installs=False,
            allow_model_downloads=False,
        ),
        runtime=ap.RuntimeResources(network=False, offline=True),
        data=ap.DataPolicy(external_egress=False),
        confirmed=True,
    )
    feasible = resources.feasibility()
    assert feasible[7]["feasible"] is True
    assert feasible[1]["feasible"] is False
    assert feasible[2]["feasible"] is False
    assert feasible[3]["feasible"] is False
    assert feasible[6]["feasible"] is False
