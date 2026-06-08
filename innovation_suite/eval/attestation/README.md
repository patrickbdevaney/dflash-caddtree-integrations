# Attestation records (evidence chain for eval runs)
configs/ exact serve config JSON · hardware/ GPU/CPU/vLLM spec · raw_outputs/ per-instance
JSONL · scores/ aggregated + Wilson CI. Reproduce from a config JSON + model weights + RULER
at the pinned commit. NOTE: commits use DCO sign-off (-s). GPG (-S) is NOT enabled — no key is
configured and auto-generating one + global commit.gpgsign was deliberately NOT done (invasive
global change). GPG was attempted with project-scoped (local, non-global) config per user permission, but is
blocked on this box (~/.gnupg was root-owned -> fixed; gpg-agent then failed: agent_genkey
'No such file or directory'). Per the user's 'if difficult, then no' instruction, signing is
**DCO sign-off (-s) only**. (Note: GPG signing would NOT encrypt results -- it only attests
authorship; the concern about encryption does not apply.) To enable -S later: fix gpg-agent
(gpgconf --launch gpg-agent) then `git -C <repo> config user.signingkey <id>; config
commit.gpgsign true` (LOCAL, not --global).
