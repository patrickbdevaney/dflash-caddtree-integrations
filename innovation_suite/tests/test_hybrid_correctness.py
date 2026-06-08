"""Stage 0 — shared correctness harness for hybrid recurrent models (GDN / Mamba-2).

HybridCorrectnessGate centralizes the three gate types used across the innovation
suite. All gates operate on lists of per-prompt token-ID lists (T=0, deterministic).

  lossless_gate : bitwise-identical T=0 required (FP8 KV scale=1.0, APC cold==warm,
                  DroPE within native context, KV offload).
  lossy_gate    : NOT bitwise; detects DEGENERATE output (repeated-token / topic
                  fixation) and reports first-divergence rate (SnapKV compression).
  recall_gate   : needle-in-haystack accuracy at multiple context lengths.

Token-ID lists are produced by tests/gen.py (saved to baseline_outputs/<tag>.json).
"""
from __future__ import annotations
from collections import Counter


class HybridCorrectnessGate:
    @staticmethod
    def lossless_gate(a: list[list[int]], b: list[list[int]]):
        """Bitwise-identical T=0. Returns (passed, diverging_indices, first_diffs)."""
        diverging, first_diffs = [], {}
        n = min(len(a), len(b))
        for i in range(n):
            if a[i] != b[i]:
                diverging.append(i)
                p = next((j for j in range(min(len(a[i]), len(b[i]))) if a[i][j] != b[i][j]),
                         min(len(a[i]), len(b[i])))
                first_diffs[i] = p
        passed = (not diverging) and len(a) == len(b)
        return passed, diverging, first_diffs

    @staticmethod
    def _degenerate(toks: list[int], tail_from: int = 20, thresh: float = 0.5) -> bool:
        """Repeated-token / topic-fixation signature: a single token dominates the tail."""
        tail = toks[tail_from:] if len(toks) > tail_from else toks
        if not tail:
            return False
        share = Counter(tail).most_common(1)[0][1] / len(tail)
        # also catch short-cycle loops (period<=4 repeating)
        cyc = any(len(tail) >= 8 and tail[k:] == tail[k - p:len(tail) - p]
                  for p in (1, 2, 3, 4) for k in (p,))
        return share > thresh or cyc

    @classmethod
    def lossy_gate(cls, base: list[list[int]], new: list[list[int]]):
        """For lossy changes. Returns (degenerate_detected, quality_delta, first_div_positions).
        Hard gate = no degenerate output. quality_delta = mean first-divergence rate vs base."""
        degenerate = [i for i, t in enumerate(new) if cls._degenerate(t)]
        first_div = {}
        for i in range(min(len(base), len(new))):
            p = next((j for j in range(min(len(base[i]), len(new[i]))) if base[i][j] != new[i][j]), -1)
            first_div[i] = p
        # quality proxy: fraction of prompts that diverge before completing
        diverged = sum(1 for v in first_div.values() if v >= 0)
        quality_delta = diverged / max(1, len(first_div))
        return (len(degenerate) > 0), quality_delta, first_div

    @staticmethod
    def recall_gate(results_by_length: dict[int, list[bool]]):
        """needle hits per context length -> {length: accuracy}."""
        return {L: (sum(hits) / len(hits) if hits else 0.0) for L, hits in results_by_length.items()}


# Trivial self-test on identical inputs (pytest-discoverable).
def test_gate_trivial_pass():
    g = HybridCorrectnessGate
    a = [[1, 2, 3], [4, 5, 6]]
    ok, div, _ = g.lossless_gate(a, [list(x) for x in a])
    assert ok and not div
    deg, _, _ = g.lossy_gate(a, a)
    assert not deg
    assert g.recall_gate({128: [True, True, False]})[128] == 2 / 3

def test_gate_catches_divergence_and_degenerate():
    g = HybridCorrectnessGate
    ok, div, _ = g.lossless_gate([[1, 2, 3]], [[1, 9, 3]])
    assert not ok and div == [0]
    deg, _, _ = g.lossy_gate([[1, 2, 3, 4]], [[7] * 40])
    assert deg
