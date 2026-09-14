"""Scheduled lane scoring (2026-09, the auto/evolve starvation recurrence).

Eligibility floors and availability buckets only rank what has been measured,
and measurement evidence EXPIRES. Thin lanes starved while serving; this pass
drives the existing probe arm so every lane carries fresh evidence. The pass
must never take the serving path down: one bad probe cannot end a lane, one
bad lane cannot end a pass.
"""
import logging

import pytest

from model_mesh.scorer import run_score_pass


class FakeRouter:
    def __init__(self):
        self.calls = []

    def probe_verdict(self, mid, op_class, messages, timeout=None):
        self.calls.append((mid, op_class))
        return ("pass", "")


CFG = {"provider": {"name": "nim"},
       "aliases": {f"auto/x{i}": {"op_class": "retain"} for i in range(2)}}
POOL = {f"auto/x{i}": [f"m{i}-{j}" for j in range(10)] for i in range(2)}


def candidates_for(index, provider, alias_cfg, alias):
    return POOL[alias]


def test_pass_probes_top_n_per_alias():
    r = FakeRouter()
    out = run_score_pass(CFG, r, candidates_for=candidates_for, top_n=6)
    assert set(out) == {"auto/x0", "auto/x1"}
    assert len(r.calls) == 12                      # 6 per alias, not 20
    assert out["auto/x0"]["m0-0"] == "pass"
    assert all(oc == "retain" for _, oc in r.calls)


def test_pass_survives_one_probe_raising():
    class Boom(FakeRouter):
        def probe_verdict(self, mid, op_class, messages, timeout=None):
            if mid.endswith("-3"):
                raise RuntimeError("dial exploded")
            return super().probe_verdict(mid, op_class, messages, timeout)
    r = Boom()
    out = run_score_pass(CFG, r, candidates_for=candidates_for, top_n=6)
    assert out["auto/x0"]["m0-3"] == "error"       # lane continues, no raise
    assert out["auto/x1"]["m1-4"] == "pass"
