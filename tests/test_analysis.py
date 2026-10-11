import pytest

from specdec.analysis import (
    SPEC_COUNTERS,
    acceptance_from_counters,
    build_table,
    expected_tokens_per_step,
    parse_counters,
    predicted_speedup,
    render_markdown,
    spec_config_arg,
)

METRICS_BEFORE = """\
vllm:spec_decode_num_drafts_total{model_name="m"} 100.0
vllm:spec_decode_num_draft_tokens_total{model_name="m"} 300.0
vllm:spec_decode_num_accepted_tokens_total{model_name="m"} 180.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="m",position="0"} 90.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="m",position="1"} 60.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="m",position="2"} 30.0
"""
METRICS_AFTER = METRICS_BEFORE.replace("100.0", "200.0").replace("300.0", "600.0").replace("180.0", "420.0") \
    .replace('position="0"} 90.0', 'position="0"} 190.0').replace('position="1"} 60.0', 'position="1"} 140.0').replace('position="2"} 30.0', 'position="2"} 90.0')


def test_parse_counters_keeps_per_position_series_separately():
    c = parse_counters(METRICS_BEFORE)
    assert c["drafts"] == 100 and c["draft_tokens"] == 300 and c["accepted_tokens"] == 180
    assert c["accepted_per_pos"] == {0: 90.0, 1: 60.0, 2: 30.0}
    assert set(SPEC_COUNTERS) == {"drafts", "draft_tokens", "accepted_tokens"}


def test_acceptance_from_counters_uses_the_delta_between_snapshots():
    a = acceptance_from_counters(parse_counters(METRICS_BEFORE), parse_counters(METRICS_AFTER))
    assert a["drafts"] == 100 and a["draft_tokens"] == 300 and a["accepted_tokens"] == 240
    assert a["acceptance_rate"] == pytest.approx(0.8)          # 240 / 300
    assert a["accepted_per_draft"] == pytest.approx(2.4)        # 240 / 100
    assert a["tokens_per_step"] == pytest.approx(3.4)           # + the bonus token the target emits
    assert a["per_position_rate"] == pytest.approx([1.0, 0.8, 0.6])  # 100/100, 80/100, 60/100


def test_acceptance_handles_no_speculation():
    a = acceptance_from_counters({}, {})
    assert a["acceptance_rate"] is None and a["tokens_per_step"] == 1.0


def test_expected_tokens_matches_geometric_formula():
    assert expected_tokens_per_step(alpha=0.5, k=1) == pytest.approx(1.5)
    assert expected_tokens_per_step(alpha=0.9, k=5) == pytest.approx((1 - 0.9**6) / 0.1)
    assert expected_tokens_per_step(alpha=1.0, k=3) == 4.0


def test_predicted_speedup_accounts_for_draft_cost():
    assert predicted_speedup(alpha=0.7, k=3, draft_cost=0.2) == pytest.approx(expected_tokens_per_step(0.7, 3) / 1.6)
    assert predicted_speedup(alpha=0.3, k=5, draft_cost=0.2) < 1.0


def test_spec_config_arg_formats_vllm_json():
    arg = spec_config_arg("draft_model", 3, model="Qwen/Qwen2.5-0.5B-Instruct")
    assert arg == ['--speculative-config', '{"method": "draft_model", "num_speculative_tokens": 3, "model": "Qwen/Qwen2.5-0.5B-Instruct"}']
    ng = spec_config_arg("ngram", 4, prompt_lookup_min=2, prompt_lookup_max=5)
    assert '"method": "ngram"' in ng[1] and '"prompt_lookup_max": 5' in ng[1]


def _run(name, k, conc, itl, tps, acc=None, tps_step=None):
    return {"config": name, "k": k, "concurrency": conc, "itl_p50_ms": itl, "output_tokens_per_s": tps,
            "acceptance": {"acceptance_rate": acc, "tokens_per_step": tps_step or 1.0}}


def test_build_table_computes_speedup_against_baseline_at_same_concurrency():
    runs = [_run("baseline", 0, 1, 30.0, 33), _run("baseline", 0, 16, 40.0, 400),
            _run("draft k=3", 3, 1, 15.0, 66, 0.7, 2.5), _run("draft k=3", 3, 16, 50.0, 320, 0.7, 2.5)]
    rows = build_table(runs)
    d1 = next(r for r in rows if r["config"] == "draft k=3" and r["concurrency"] == 1)
    d16 = next(r for r in rows if r["config"] == "draft k=3" and r["concurrency"] == 16)
    assert d1["itl_speedup"] == 2.0 and d1["throughput_ratio"] == 2.0
    assert d16["itl_speedup"] == 0.8 and d16["throughput_ratio"] == 0.8
    assert next(r for r in rows if r["config"] == "baseline" and r["concurrency"] == 1)["itl_speedup"] == 1.0


def test_render_markdown_one_row_per_run():
    md = render_markdown(build_table([_run("baseline", 0, 1, 30.0, 33), _run("ngram k=4", 4, 1, 25.0, 40, 0.4, 1.6)]))
    assert md.count("\n| ") == 2 and "acceptance" in md
