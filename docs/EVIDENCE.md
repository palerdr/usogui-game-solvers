# Air Poker Evidence Ledger

## Purpose

This file records what the source establishes before the software interprets it.
It is not a rules engine, a strategy guide, or a place to resolve ambiguity by
intuition. `MODEL.md` converts this evidence into explicit solver assumptions.

## Evidence standard

Evidence is classified as:

| Class | Meaning |
|---|---|
| `P` | Primary: directly depicted or stated in the manga chapter cited. |
| `D` | Derived: a necessary mathematical consequence of primary facts. |
| `I` | Interpretation: a plausible reading not uniquely forced by the source. |
| `S` | Secondary: commentary in `Air-Poker-Summary.md`; useful for navigation, not authority. |

Confidence describes the repository's present audit, not the truth of the manga.
`high` means the rule is directly supported and unambiguous. `medium` means the
result is supported but timing or a counterfactual remains open. `open` means the
solver must choose and label an assumption.

## Mechanical evidence

| ID | Class | Chapters | Source observation | Consequence | Confidence |
|---|---:|---:|---|---|---:|
| E01 | P | 429–431 | Each lower player begins with 25 Air-BIOS. A BIOS is both an air supply and a betting unit. | Chips and survival resources are coupled. | high |
| E02 | P | 429–431 | One opened BIOS contains about five minutes of air and cannot be wagered. | The engine separates sealed BIOS from residual seconds in the open BIOS. | high |
| E03 | P | 430 | The match has at most five poker rounds. | The horizon is finite. | high |
| E04 | P | 430 | Antes rise by round: 1, 2, 3, 4, 5 BIOS per player. | The ante schedule is deterministic. | high |
| E05 | P | 430–431 | Each lower player selects one unused metal card; both selected numbers become known before betting. | Number choice precedes a one-street betting game. | high |
| E06 | P | 431 | Betting order, action timing, wager limits, and defeat for inability to pay are stated. | Legal actions depend on contributions and sealed BIOS. | high |
| E07 | P | 436–440 | An upper player constructs five ordinary cards whose Air Poker values sum to the lower player's metal number; ace is 1 and king is 13. | A legal hand is a five-card subset with the target rank sum. | high |
| E08 | P | 436–440 | Constructed exact cards leave one shared virtual deck. | Later hand support depends on earlier exact-card use. | high |
| E09 | P/D | 436–440 | A standard deck has 52 exact cards; ten constructed hands nominally require 50 card selections. | Suits distinguish cards even though suits do not affect the target sum. | high |
| E10 | P | 444 | If the two judged hands reuse an exact card, calamity penalizes the loser. | Overlap is legal at construction and is resolved at judgement. | high |
| E11 | P | 448–456 | A hand may be no-rank; double no-rank causes another construction attempt and reveals additional information. | `None` must be a chance outcome and retry must be modeled explicitly. | high |
| E12 | P | 448–456 | Witch Pain and the invalid fourth-round construction affect upper-player behavior. | Human construction is not faithfully represented by a uniform policy. | high |
| E13 | P | 466–467 | The fifth poker round is followed by the physical survival result; Baku survives and Lalo dies. | Match utility ultimately concerns survival, not poker points alone. | high |

## Canonical instance

| ID | Class | Source | Fact used by the repository | Confidence |
|---|---:|---|---|---:|
| E14 | P/S | Air Poker arc; secondary summary | Baku's metal cards are `25, 26, 36, 39, 45`. | high |
| E15 | P/S | Air Poker arc; secondary summary | Lalo's metal cards are `8, 15, 44, 47, 63`. | high |
| E16 | D | E14–E15 | The ten values are distinct and each lies in the feasible five-card sum range `6..64`. | high |
| E17 | S | `Air-Poker-Summary.md` | The depicted order is `36/15`, `39/8`, `45/47`, `26/63`, `25/44`. | medium |
| E18 | S | `Air-Poker-Summary.md` | Round 3 reaches judgement, Lalo's 47 beats Baku's 45, and overlap produces calamity. | medium |
| E19 | S | `Air-Poker-Summary.md` | Baku folds Round 4; Baku's 25 beats Lalo's 44 in Round 5. | medium |

The secondary summary is retained as a navigation aid. Exact page and panel
references should be added before a disputed rule is promoted from `S` or `I`.

## Mathematical consequences

These facts require no behavioral interpretation.

| ID | Premises | Consequence |
|---|---|---|
| D01 | E07, four suits per rank | The minimum legal sum is `A+A+A+A+2 = 6`; the maximum is `K+K+K+K+Q = 64`. |
| D02 | E07–E09 | Legal-hand support is `L(D,n) = {H ⊆ D : |H|=5, sum(H)=n}`. |
| D03 | E08, E10 | Both current hands must be validated against a common pre-round deck; otherwise exact-card overlap could not occur. |
| D04 | E08 | After a resolved pair `(H0,H1)`, the next deck is `D − (H0 ∪ H1)`. |
| D05 | E03–E04 | Total mandatory ante exposure per player is 15 BIOS if all five rounds begin. |
| D06 | E01–E02 | A strategy can win chips yet lose the physical match; terminal utility cannot be round poker value. |
| D07 | E05, hidden current choice | The game has imperfect information even if the canonical metal-card allocation is fixed. |
| D08 | E08 | Hidden exact hands create a hidden, path-dependent remaining deck and correlate later chance outcomes. |

## Evidence gaps

The source does not uniquely determine the following counterfactuals. They are
model decisions, not facts.

| Gap | What is not established | Required model decision |
|---|---|---|
| G01 | The exact stochastic rule used by Hal and Fukurou when several legal hands exist. | Define an upper construction policy. |
| G02 | Whether every upper construction is fully committed before lower betting in a generic counterfactual. | Fix chance timing. |
| G03 | Whether calamity would be applied after a fold, when hands are not judged. | Fix fold settlement. |
| G04 | Continuous oxygen consumption, individual lung capacity, and action-duration effects. | Choose a survival-state abstraction. |
| G05 | A probability distribution over unseen metal-card allocations or residual air. | Choose a fixed conditional root or an explicit root prior. |
| G06 | How accidental failure, deliberate no-rank, memory, communication, and Witch Pain change construction probabilities. | Define separate behavioral models; do not merge causes silently. |
| G07 | What lower players observe about exact upper hands, hand categories, failure causes, and the remaining deck outside depicted events. | Freeze the lower information partition. |
| G08 | What happens if a retry policy has probability one of another double no-rank. | Reject the model or define an evidenced fallback. |
| G09 | The utility of pain, time, regret, or manner of death relative to binary survival. | Fix terminal utility. |

## Source-to-code trace

| Evidence | Executable location | Regression coverage |
|---|---|---|
| E03–E06 | `src/engine/game.py` | `src/test/test_game.py` |
| E07, D01–D02 | `src/engine/poker_math.py` | exhaustive tests in `src/test/test_poker_math.py` |
| E08, E10–E11, D03–D04 | `src/engine/game.py` | shared-deck, retry, no-rank, and calamity tests |
| E14–E16 | `src/engine/game.py` constants and `initial_state` | initial-state representation tests |
| G02–G09 | `docs/MODEL.md` | model-lock and solver-interface tests required by `BUILD.md` |

## Change rule

Every mechanical change must cite an evidence ID or create a new one. If the
support is interpretive, the change belongs first in `MODEL.md`. Tests establish
that software follows the selected model; they do not upgrade an assumption into
source evidence.
