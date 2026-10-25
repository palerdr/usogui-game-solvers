# Arena Project Instructions

`src/arena/` is the game library that the browser app uses. It may import
public interfaces from `stl` and `dth`, and those projects must not import it.

- The STL engine remains the only canonical live-game referee.
- The completed DTH tablebase is the default Hal policy provider.
- Providers return policy distributions; `PolicyDrivenAgent` alone masks and
  samples literal legal seconds.
- [`src/browser/`](../browser/README.md) is the browser game,
  `python -m browser`, with the hosted Vercel deployment. Its README holds the
  server, hosted-session, and deploy invariants.
- DTH projection is exact for the shared state and actions 1..60. The only
  prospective mismatch is Baku's legal Dropper action 61 in the public leap
  window; arena keeps that canonical action even though DTH has no 61 policy.
- Projection adapters may not alter canonical game state or transitions.
- Keep generated artifacts in the owning project, never under `src/arena/`.
- `policies/` holds the two Hal providers. `perfect_hal.py` is the base
  opponent model and `translated_hal.py` the production translated Hal, which
  `translated_hal_adapter.py` wraps with the exact leap-window fallback.
  `config/translated_hal_v1_selection.json` holds its selected values.
- `policies/registry.py` builds Hal for the browser app: `exact` (the
  certified DTH equilibrium) or `translated`.
- `policies/__init__.py` imports nothing. Import each provider from its own
  module. `src/browser/tests/test_runtime_imports.py` checks this in
  a fresh interpreter.
- `variants.py` holds `PureDTHGame`. `contracts.py`, `agent.py`, `session.py`,
  and `variants.py` import no project except `stl.engine`; `dth` enters
  through the adapters and `policies/`.
- `transcript.py` writes the public play transcript file for the browser's
  `--transcript`. `session.py` stays free of I/O.
- `testing.py` holds the fakes that the tests of more than one project share,
  such as `StageAgent` and `make_session`. No runtime module imports it.
- No arena module imports `browser`.

## Play surfaces and the session

`session.py` owns the phase machine every interactive surface drives:
`RULES -> AWAITING_ACTION -> AWAITING_ACK -> GAME_OVER`. It performs no I/O and
no rendering; it only sequences the referee calls. `src/browser/app.py` is a
thin adapter over it.

Hal's action is chosen inside `PlaySession.submit`, after the human's second has
been accepted and validated. That ordering is the hidden-information guarantee,
not a convenience: while a client is deciding, Hal's second does not exist in
the process, so no snapshot can leak it. Do not hoist that call earlier to
"prepare" a move.

Engine identities remain exactly `Hal` and `Baku`. A display label never
replaces Baku's rule-bearing identity, and `Hal` is reserved as a display
label.

The Hal research record is in git history at commit `938fb25`, under
`src/hal_lab/docs/HAL_RESEARCH.md`.
