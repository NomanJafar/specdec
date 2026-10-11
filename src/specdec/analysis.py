"""Acceptance-rate bookkeeping and the comparison table for speculative decoding runs.

vLLM exports cumulative counters; acceptance for one measurement window is the delta
between a snapshot before and after the load (CONCEPTS.md section 4).
"""

from __future__ import annotations

import json
import re

SPEC_COUNTERS = {
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
}
_PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
_LINE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(\S+)")
_POS = re.compile(r'position="(\d+)"')


def parse_counters(metrics_text: str) -> dict:
    out: dict = {"accepted_per_pos": {}}
    for line in metrics_text.splitlines():
        m = _LINE.match(line)
        if not m or line.startswith("#"):
            continue
        name, labels, value = m.group(1), m.group(2) or "", float(m.group(3))
        for short, full in SPEC_COUNTERS.items():
            if name == full:
                out[short] = out.get(short, 0.0) + value
        if name == _PER_POS:
            pm = _POS.search(labels)
            if pm:
                pos = int(pm.group(1))
                out["accepted_per_pos"][pos] = out["accepted_per_pos"].get(pos, 0.0) + value
    return out


def acceptance_from_counters(before: dict, after: dict) -> dict:
    d = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in SPEC_COUNTERS}
    per_pos = {p: after.get("accepted_per_pos", {}).get(p, 0.0) - before.get("accepted_per_pos", {}).get(p, 0.0)
               for p in sorted(after.get("accepted_per_pos", {}))}
    drafts, draft_tokens, accepted = d["drafts"], d["draft_tokens"], d["accepted_tokens"]
    return {
        **d,
        "acceptance_rate": accepted / draft_tokens if draft_tokens else None,
        "accepted_per_draft": accepted / drafts if drafts else None,
        # each verify step yields accepted drafts plus the one token the target always produces
        "tokens_per_step": 1.0 + (accepted / drafts if drafts else 0.0),
        "per_position_rate": [per_pos[p] / drafts for p in per_pos] if drafts else [],
    }


def expected_tokens_per_step(alpha: float, k: int) -> float:
    if alpha >= 1.0:
        return float(k + 1)
    return (1 - alpha ** (k + 1)) / (1 - alpha)


def predicted_speedup(alpha: float, k: int, draft_cost: float, overhead: float = 0.0) -> float:
    return expected_tokens_per_step(alpha, k) / (1 + k * draft_cost + overhead)


def spec_config_arg(method: str, num_speculative_tokens: int, **extra) -> list[str]:
    cfg = {"method": method, "num_speculative_tokens": num_speculative_tokens, **extra}
    return ["--speculative-config", json.dumps(cfg)]


def build_table(runs: list[dict]) -> list[dict]:
    base = {r["concurrency"]: r for r in runs if r["config"] == "baseline"}
    rows = []
    for r in runs:
        b = base.get(r["concurrency"])
        acc = r.get("acceptance") or {}
        rows.append({
            "config": r["config"], "k": r.get("k", 0), "concurrency": r["concurrency"],
            "acceptance_rate": acc.get("acceptance_rate"), "tokens_per_step": acc.get("tokens_per_step"),
            "itl_p50_ms": r.get("itl_p50_ms"), "ttft_p50_ms": r.get("ttft_p50_ms"),
            "output_tokens_per_s": r.get("output_tokens_per_s"),
            "itl_speedup": round(b["itl_p50_ms"] / r["itl_p50_ms"], 2) if b and r.get("itl_p50_ms") else None,
            "throughput_ratio": round(r["output_tokens_per_s"] / b["output_tokens_per_s"], 2)
            if b and r.get("output_tokens_per_s") and b.get("output_tokens_per_s") else None,
        })
    return rows


def _f(x, d=1, suffix=""):
    return "–" if x is None else f"{x:.{d}f}{suffix}"


def render_markdown(rows: list[dict]) -> str:
    head = ["config", "concurrency", "acceptance rate", "tokens / step", "ITL p50", "ITL speedup", "out tok/s", "throughput vs baseline"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        lines.append("| " + " | ".join([
            r["config"], str(r["concurrency"]), _f(r["acceptance_rate"], 2), _f(r["tokens_per_step"], 2),
            _f(r["itl_p50_ms"], 1, " ms"), _f(r["itl_speedup"], 2, "x"), _f(r["output_tokens_per_s"], 0), _f(r["throughput_ratio"], 2, "x"),
        ]) + " |")
    return "\n".join(lines) + "\n"
