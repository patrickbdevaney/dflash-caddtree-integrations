# NVFP4 expert weight-loader patch for dllm-plugin `dllm_plugin/models/llada2.py`.
#
# WHAT: extends LLaDA2ForCausalLM.load_weights (phase-2 per-expert loop) to load
#       compressed-tensors NVFP4 expert weights, not only BF16.
# WHY:  the NVFP4 model stores each expert projection as 4 tensors
#       (weight_packed / weight_scale / weight_global_scale / input_global_scale);
#       the original loader looked only for plain `.weight` and raised
#       "Missing weights for layer 1 expert 0". With this patch the NVFP4 model
#       loads and vLLM selects the CUTLASS FP4 grouped-MoE kernel on sm_110.
#
# Mapping: gate_proj -> w13_* (shard w1), up_proj -> w13_* (shard w3),
#          down_proj -> w2_* (shard w2). FusedMoE NVFP4 param name = prefix + suffix
#          (w13_weight_packed, w13_weight_scale, w13_weight_global_scale,
#           w13_input_global_scale, and the w2_ equivalents).
#
# This is the block that REPLACES the original BF16-only
# "gate_weight = expert_params.get('gate_proj.weight') ... weight_loader(...)" body,
# inside `for expert_id in range(self.num_experts):`.

# ---- replacement body (per expert) ----
experts = layer.mlp.experts
_shards = (("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2"))

if "gate_proj.weight" in expert_params:
    # BF16 path (unchanged behavior)
    for proj, shard in _shards:
        w = expert_params.get(f"{proj}.weight")
        if w is None:
            raise ValueError(
                f"Missing {proj}.weight for layer {layer_id} expert {expert_id}"
            )
        pname = "w2_weight" if shard == "w2" else "w13_weight"
        param = getattr(experts, pname)
        wl = getattr(param, "weight_loader", default_weight_loader)
        wl(param, w, f"{proj}.weight", shard_id=shard, expert_id=expert_id)
elif "gate_proj.weight_packed" in expert_params:
    # NVFP4 compressed-tensors path: packed weight + block scale + 2 global scales
    _suffixes = (
        "weight_packed", "weight_scale",
        "weight_global_scale", "input_global_scale",
    )
    for proj, shard in _shards:
        prefix = "w2_" if shard == "w2" else "w13_"
        for sfx in _suffixes:
            w = expert_params.get(f"{proj}.{sfx}")
            if w is None:
                continue
            param = getattr(experts, prefix + sfx, None)
            if param is None:
                continue
            wl = getattr(param, "weight_loader", default_weight_loader)
            wl(param, w, f"{proj}.{sfx}", shard_id=shard, expert_id=expert_id)
else:
    raise ValueError(
        f"Missing weights for layer {layer_id} expert {expert_id}: "
        f"keys={sorted(expert_params)[:6]}"
    )
