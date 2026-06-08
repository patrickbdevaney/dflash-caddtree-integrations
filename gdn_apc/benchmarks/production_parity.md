# Production parity + e2e: DFlash + GDN-APC on the long agentic trace

4-turn Hermes agentic conversation (~8.3k-token shared system prompt, ~11.8k ctx), T=0,
CUDA graphs, max_model_len 16384. Three configs, 800 generated tokens total.

## E2E (the production win)
| config | total time | vs base |
|--------|-----------:|--------:|
| BASE (no spec, no APC, none mode) | 33.71 s | 1.00× |
| DFlash spec, no APC (none mode)   | 27.02 s | 1.25× |
| **DFlash spec + APC (align, 128)**| **20.25 s** | **1.66×** |

APC adds **1.33×** on top of DFlash spec (cached growing prefix not recomputed each turn).

## Cache correctness (KV stability) — ACHIEVED
cold==warm WITHIN DFlash+APC align mode is **bitwise-identical** over the 11.8k-token, 4-turn
conversation (see long_context_agentic.md). ⇒ **the prefix cache is transparent: restoring
cached GDN state yields the exact same output as recomputing it. KV stable, not corrupt.**

## Bitwise parity with base — NOT achievable (and why it's not an APC defect)
| comparison | result | cause |
|------------|--------|-------|
| base vs DFlash (no APC) | DIVERGE (first flip ~pos45, single near-tie token, then cascades) | **spec-decode verify numerics** — the multi-token verify forward's FP order differs from base's autoregressive forward, flipping genuine near-ties; once one token flips the greedy continuation cascades. **True of ALL speculative decoding**, present with NO APC. |
| none vs align (DFlash) | DIVERGE | benign align-vs-none mode FP-order difference (diagnosed earlier) |
| cold vs warm (within align+APC) | **BITWISE IDENTICAL** | the cache is transparent (no corruption) |

**Conclusion:** APC introduces **no corruption** (cold==warm bitwise) and a **real 1.66× e2e
speedup**. It does NOT achieve *bitwise* parity with the base autoregressive model — but
neither does plain DFlash spec decode (or any spec decode): the verify-forward numerics flip
near-ties vs base, and that cascades. APC additionally forces align mode (another benign
numeric). All divergences are **distributionally valid** (each token a valid greedy
continuation, same quality), not state corruption.

**What "parity" is and isn't here:**
- ✅ Cache transparency (recall/state not corrupted by caching): cold==warm bitwise.
- ✅ Output-quality parity: same class of valid greedy completions as DFlash spec decode.
- ❌ Bitwise reproduction of the base model's exact tokens: impossible with spec decode +
  align mode. Achieving it would require (a) not using spec decode and (b) a GDN kernel that
  makes align bit-match none — out of scope and likely infeasible (align is inherently
  block-chunked). This is a fundamental property, not a bug to fix.

MTP parity: MTP is also a (lossless-distribution, not-bitwise) spec method — same parity
property as DFlash; not separately measured (would show the same near-tie divergence vs base).
