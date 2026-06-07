"""Shared in-process test harness for GDN tree-speculation invariant tests.

Confirmed environment facts (harness probe, vllm-dflash-thor:fa-native):
  - VLLM_ENABLE_V1_MULTIPROCESSING=0 -> UniProcExecutor, worker in-process.
  - The spec GDN recurrent update goes through
    vllm.model_executor.layers.fla.ops.fused_sigmoid_gating
        .fused_sigmoid_gating_delta_rule_update(...)
    with kwargs: initial_state [num_blocks,32,128,128] (per-layer ssm_state pool),
    ssm_state_indices [batch, num_spec+1], num_accepted_tokens [batch],
    inplace_final_state (bool). Returns (out, final_state).
  - Each GDN layer owns a distinct ssm_state pool -> distinguish layers by
    initial_state.data_ptr().

Tree-width control is hardcoded during dev (the tree code reads a module global
set by build_llm(tree_width=...) once it exists). Until tree code lands,
tree_width is recorded for bookkeeping only; W=1 == current linear DFlash.
"""
import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "1")
os.environ.setdefault("VLLM_MARLIN_USE_ATOMIC_ADD", "1")

import torch

FIXED_PROMPTS = [
    "Write a haiku about the ocean.",
    "List the first 8 prime numbers.",
    "Explain what a binary search tree is in two sentences.",
    "Translate 'good morning' into French, Spanish, and German.",
    "What is the capital of Japan?",
    "Write a Python function that reverses a string.",
    "Summarize the water cycle in one sentence.",
    "Name three primary colors.",
    "What comes next: 2, 4, 8, 16, ...?",
    "Describe the taste of a lemon in one word.",
]


def build_llm(tree_width: int = 1, max_model_len: int = 4096, max_num_seqs: int = 1,
              attention_backend: str | None = None):
    """Construct the 35B-A3B + DFlash draft engine in-process (eager, deterministic).

    tree_width is threaded to the tree code via env so a single image can
    exercise W=1/2/3. W=1 hits the unchanged linear path. attention_backend
    forces the target backend (FLEX_ATTENTION needed for the DDTree ancestor
    mask at W>1); None keeps the default (FLASH_ATTN) for the linear baseline.
    """
    os.environ["DFLASH_TREE_WIDTH"] = str(tree_width)
    nspec = int(os.environ.get("DFLASH_NUM_SPEC", "12"))
    from vllm import LLM
    kw = dict(
        model="/model", tokenizer="/model",
        quantization="compressed-tensors", kv_cache_dtype="auto",
        speculative_config={"method": "dflash", "num_speculative_tokens": nspec, "model": "/drafter"},
        gpu_memory_utilization=0.78, max_model_len=max_model_len, max_num_seqs=max_num_seqs,
        enforce_eager=(os.environ.get("DFLASH_GRAPHS") != "1"), trust_remote_code=True,
        hf_overrides={"architectures": ["Qwen3_5MoeForConditionalGeneration"]},
    )
    if os.environ.get("DFLASH_GRAPHS") == "1":
        kw["compilation_config"] = {"cudagraph_capture_sizes": [1, 13]}
    if attention_backend is not None:
        kw["attention_backend"] = attention_backend
    return LLM(**kw)


def greedy(llm, prompts, max_tokens=32, seed=0):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=seed)
    outs = llm.generate(prompts, sp)
    return [list(o.outputs[0].token_ids) for o in outs]


def _fp(state_slot: torch.Tensor):
    """Cheap fingerprint of a [32,128,128] state slot: (l2norm, sum)."""
    f = state_slot.float()
    return (float(f.norm().item()), float(f.sum().item()))


class StateCapture:
    """Monkeypatch the spec GDN kernel to record per-call recurrent-state info.

    For the FIRST ssm_state pool (one GDN layer) seen, capture per call:
      - ssm_state_indices row for sequence 0, num_accepted_tokens[0]
      - pre/post fingerprints of every referenced slot (pre from initial_state,
        post from the returned final_state)
      - inplace flag, and whether the canonical/seed slot (num_accepted-1) was
        modified (the INVARIANT-3 probe).
    Capture is bounded to `max_calls` to keep overhead sane.
    """
    def __init__(self, max_calls=120, target_seq=0):
        self.max_calls = max_calls
        self.target_seq = target_seq
        self.records = []
        self._layer_ptr = None
        self._orig = None

    def __enter__(self):
        import vllm.model_executor.layers.fla.ops.fused_sigmoid_gating as fsg
        try:
            import vllm.model_executor.layers.mamba.gdn_linear_attn as gla
        except Exception:
            gla = None
        self._fsg = fsg
        self._gla = gla
        self._orig = fsg.fused_sigmoid_gating_delta_rule_update

        def wrap(*a, **kw):
            rec = None
            try:
                ssi = kw.get("ssm_state_indices")
                init = kw.get("initial_state")
                nat = kw.get("num_accepted_tokens")
                if ssi is not None and init is not None and len(self.records) < self.max_calls:
                    ptr = init.data_ptr()
                    if self._layer_ptr is None:
                        self._layer_ptr = ptr
                    if ptr == self._layer_ptr:
                        row = ssi[self.target_seq].tolist()
                        na = int(nat[self.target_seq].item()) if nat is not None else 1
                        pre = {int(s): _fp(init[int(s)]) for s in row if int(s) > 0}
                        rec = {
                            "indices": row,
                            "num_accepted": na,
                            "inplace": bool(kw.get("inplace_final_state", True)),
                            "pre": pre,
                        }
            except Exception as e:
                rec = {"capture_err": repr(e)}
            ret = self._orig(*a, **kw)
            try:
                if rec is not None and "capture_err" not in rec:
                    final = ret[1] if isinstance(ret, (tuple, list)) and len(ret) > 1 else kw.get("initial_state")
                    post = {}
                    for s in rec["indices"]:
                        s = int(s)
                        if s > 0 and s < final.shape[0]:
                            post[s] = _fp(final[s])
                    rec["post"] = post
                    seed_idx = int(rec["indices"][rec["num_accepted"] - 1])
                    init = kw.get("initial_state")
                    rec["seed_idx"] = seed_idx
                    # did the canonical/seed slot change in the pool after the call?
                    if seed_idx > 0:
                        rec["seed_post_pool"] = _fp(init[seed_idx])
                    self.records.append(rec)
            except Exception as e:
                self.records.append({"post_err": repr(e)})
            return ret

        fsg.fused_sigmoid_gating_delta_rule_update = wrap
        if gla is not None and hasattr(gla, "fused_sigmoid_gating_delta_rule_update"):
            gla.fused_sigmoid_gating_delta_rule_update = wrap
        return self

    def __exit__(self, *exc):
        self._fsg.fused_sigmoid_gating_delta_rule_update = self._orig
        if self._gla is not None and hasattr(self._gla, "fused_sigmoid_gating_delta_rule_update"):
            self._gla.fused_sigmoid_gating_delta_rule_update = self._orig
        return False
