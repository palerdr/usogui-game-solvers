# Hal: two live providers, one research record

`src/stl/hal.py` owns Exact and Translated Hal. `make_hal(kind, table)` accepts
`exact` or `translated`; the hosted game always serves Translated. Both use
`stl.dth.Table` for the certified pure-DTH stage and continuation.

Exact returns its equilibrium. Translated scores a mixture of public-history
experts, including copy and mirror offsets from Hal's prior actions, and
best-responds to the forecast in that stage. Its 13 selected settings are
inline defaults. Expert and offset retention are 0.97; learning rate is 1.25.
It is an empirical exploiter with no full-game safety or human-win-rate claim.

The model scores predictions before the reveal and updates afterward. At a
leap turn it uses DTH equilibrium, leaves Baku's action 61 legal, skips the
reveal in its 60-action model, and clears sequence references while retaining
prior evidence. No provider claims to optimize action 61.

Hosted opponent memory survives across games through its serialized checkpoint.
Command-log replay and compare-and-set govern accepted state; a changed code
version ends old sessions. `tests/hal_trace.json` pins the translated decision
trace, including leap turns and a memory round trip. Server tests cover
checkpoint replay and isolation.

## Results retained from the retired studies

Every row below is recoverable from `938fb25:src/hal_lab/docs/HAL_RESEARCH.md`
and its linked compact records. `938fb25` is the archival commit, not a claim
that all studies ran at that revision. Registered execution commits are named
where recorded. The retired implementations and trainers stay in git history.

| Approach | Recorded result | Decision / commit |
| --- | --- | --- |
| Exact | Equilibrium benchmark; 0.4955 score in the 112-game PM confirmation | Keep live; `938fb25` |
| Adaptive | 0.7188 score in the same PM comparison | Retired; `938fb25` |
| Exploit (epsilon-menu actor-critic) | Candidate worst-case loss independently certified; no retained result here establishes promotion | Retired; `938fb25` |
| Aggro recurrent tactical policy | 0.7232 score in PM confirmation; memory-capability warm start failed all four role/mode cells | Retired; `938fb25` |
| Perfect v1 / Old | 107 wins, 5 losses in PM confirmation (0.9554) | Base expert model retained inside Translated; `938fb25` |
| PM v3 | 97 wins, 14 losses, 1 stop; score 0.8705, below Old; PM-minus-Old -0.0848 [-0.1518,-0.0223] | Failed promotion; executed `b76147f`, archived `938fb25` |
| Bayesian Perfect v2 | 677/768 synthetic wins vs Old 675; gain 0.26 points [-1.56,2.08]; human-fitted 114/128 vs 120 | No established improvement; retired; `938fb25` |
| Adaptive policy ensemble | 1,355/1,536 synthetic wins vs Old 1,370; human-fitted 465/512 vs 473 | No established improvement; retired; `938fb25` |
| Fixed 50/50 ensemble | 1,362/1,536 synthetic wins; human-fitted 465/512 | Retired; `938fb25` |
| Translated v1 | Confirmation 7,493/8,192 vs Old 7,320; +2.11 points [1.55,2.66]. Human-fitted 4,623/5,120 vs 4,617; interval [-0.33,0.61] | Keep live; first smaller holdout's uncertainty gate failed and remains part of the record; `938fb25` |
| External repetition prior | 913/1,024 synthetic wins vs Translated 914; both 190/192 offset stress | No clear gain; retired; `938fb25` |
| Neural selector pilot | 555/608 league wins vs Translated 541; exploratory +2.30 points [0.33,4.28] | No deployment; `938fb25` |
| Transformer / GRU predictor pilots | 553/608 and 552/608 wins; corrected transformer history audit 550 full, 550 shuffled, 552 cleared | No neural-history benefit established; retired; `938fb25` |
| Neural adversary / session-reward actor | Adversary 90/256 vs untrained 95; session actor 462/608 vs immediate control 545 | Proposed advantages unsupported; retired; `938fb25` |
| Selector architecture study | 2,432-game comparison: retrained small MLP 2,232 wins, residual MLP 2,241, selected attention 2,235, Translated 2,198 | Research only; `938fb25` |
| Exploit continuation | Selected `la_role_g025` failed P1, P3, P5; P4 and P6 passed | Keep Translated; protocol `b8f87b4`, archived `938fb25` |
| Hal predictability | State-only next-action accuracy 67% vs equilibrium 22%; fixed counter won 26/500 vs exact opponent 252 | Predictability did not establish successful exploitation; `938fb25` |
| Leap support prediction | Exact support match 72.8% on H1_44 and 20.8% on H2_39; required 90% on each | Failed; revision 2 frozen; `938fb25` |

Synthetic opponents, emulators fitted to eight recorded browser identities,
and logged-state one-step values do not establish a gain against new humans.
No retired experiment's threshold or failed result has been rewritten here.
The historical paper [pm_hal.tex](paper/pm_hal.tex) still describes retired
local providers; its claims await the owner's review.
