# Air Poker Model Contract

## Objective

The solver targets one finite, two-player, zero-sum extensive-form game. This
file freezes every choice not forced by `EVIDENCE.md`. A result is meaningful
only for the model identifier and fingerprint that produced it.

The first target is:

```text
model_id = canonical-uniform-v1
```

It is a mechanical baseline, not a claim that Hal or Fukurou behaved uniformly.

## Authority

1. `EVIDENCE.md` classifies source facts and gaps.
2. This file resolves those gaps for a named model.
3. `ENGINE.md` and `src/engine/` implement the locked mechanics.
4. `SOLVER.md` may optimize traversal but may not alter the game.
5. `BUILD.md` defines delivery order and acceptance gates.

Conflicts stop the build. Solver convenience never overrides an earlier layer.

## Formal object

The model is an extensive-form game

\[
\Gamma=(N,H,Z,A,P,\sigma_c,\mathcal I,u),
\]

where `N={0,1}` are the lower teams, `H` are legal histories, `Z` are terminal
histories, `A(h)` are engine actions, `P(h)` is the acting lower player or chance,
`sigma_c` is the locked upper construction kernel, `I` is the lower information
partition, and `u:Z→{(-1,1),(0,0),(1,-1)}` is terminal survival utility.

## Locked baseline

| ID | Decision | Lock | Evidence/gap |
|---|---|---|---|
| M01 | Strategic agents | Only the two lower teams choose strategically. Upper participants are fixed chance policies. | G01 |
| M02 | Root | Use Baku `(25,26,36,39,45)` versus Lalo `(8,15,44,47,63)` with residual air `(300,300)`. | E14–E16, G05 |
| M03 | Root interpretation | This is a conditional analysis of the canonical configuration. The fixed allocation is part of the common game definition; it is not a learned prior over unknown deals. | G05 |
| M04 | Initial resources | Start after each player opens one BIOS and pays the Round-1 ante: 23 sealed BIOS plus the residual seconds. | E01–E04 |
| M05 | Upper baseline | Each upper policy is uniform over exact legal five-card masks in `L(D,n)`; impossible targets return no-rank. | G01 |
| M06 | Upper independence | Before retry, upper policies factor conditionally on the same pre-round deck and their own targets. | G01, D03 |
| M07 | Upper observation | Each upper policy sees the pre-round deck and its own target. It sees the opposing target only on retry. | E11, G06 |
| M08 | Chance timing | Both upper hands commit before lower betting and remain hidden from lower players. | G02 |
| M09 | Shared deck | Both hands are sampled against the same pre-round deck; their exact union is removed after resolution. | D03–D04 |
| M10 | Double no-rank | The first double no-rank enters retry. A stationary retry kernel is conditioned on eventual non-double-no-rank resolution. If the retry has zero resolving mass, the round ends with no judgement: both antes return, no card leaves the deck, no calamity, no winner, and play continues with the next ante or the final survival comparison. | E11, G08 |
| M11 | Fold | A fold settles immediately without calamity; committed constructed cards remain removed. | G03 |
| M12 | Breathing | Continuous consumption, action duration, standing, and physiology are omitted. Residual seconds are fixed inputs. | G04 |
| M13 | Final survival | After Round 5, compare `sealed_bios*300 + remaining_air_seconds`. Higher wins; equality ties. | E02, E13, G04 |
| M14 | Utility | Win/loss/tie returns are `+1/-1/0`; pain, regret, and time are excluded. | E13, G09 |
| M15 | Numeric actions | Check, call, fold, and every legal integer raise are distinct actions. No betting abstraction is active. | E06 |
| M16 | Hand ranking | Standard exact five-card poker ordering with suits relevant only to flushes and exact-card identity. | E07–E10 |

## Information lock

A lower policy may depend only on its `InformationState`. At a decision it knows:

- its own sealed BIOS, residual seconds, and unused metal cards;
- the opponent's public sealed-BIOS count and count of unused metal cards;
- its own current number;
- both numbers only after both choices close;
- public betting state;
- completed public round records: numbers, betting actions, winner/tie, calamity.

It does not receive:

- either exact upper hand;
- the exact remaining deck;
- the opponent's residual seconds;
- an RNG state or future chance result;
- an omniscient explanation of no-rank;
- any full `GameState` field not projected into `InformationState`.

The fixed root in M03 deserves care: omitting a constant from an information
record does not create Bayesian uncertainty about it. A future private-deal model
must add an explicit root chance distribution; it must not merely hide root fields.

## Chance kernel

For deck `D` and target `n`, define

\[
L(D,n)=\{H\subseteq D: |H|=5,\ \sum_{c\in H}v(c)=n\}.
\]

The baseline marginal policy is

\[
q(H\mid D,n)=
\begin{cases}
1/|L(D,n)| & H\in L(D,n),\\
1 & H=\bot\text{ and }L(D,n)=\varnothing,\\
0 & \text{otherwise.}
\end{cases}
\]

Before retry, the joint kernel is

\[
K(H_0,H_1\mid D,n_0,n_1)=q(H_0\mid D,n_0)q(H_1\mid D,n_1).
\]

The consequences are not independent: overlap determines calamity and the union
determines the next deck. A sampler must therefore return the complete pair, not
sample winner, overlap, deck removal, or calamity separately.

On retry, exclude `(⊥,⊥)` and divide remaining mass by
`1-K(⊥,⊥)`. If that denominator is zero, the retry has the single outcome
`(⊥,⊥)` with probability one and the round settles with no judgement (M10).
The lower players observe it as a completed round with no winner and no
calamity. Because the ten metal cards are distinct, equal poker rank is
impossible, so a recorded round without a winner is always a no-judgement round.

## State and transition lock

The authoritative transition order is:

```text
pay ante
→ Team 0 chooses privately
→ Team 1 chooses without observing Team 0's choice
→ reveal both numbers
→ draw one complete upper HandPair
→ remove the exact-card union
→ play one betting street
→ settle fold or judgement and possible calamity
→ pay the next ante or resolve terminal survival
```

The engine owns legality and transitions. A solver calls the engine; it does not
reimplement pot size, deck depletion, retry, calamity, observations, or returns.

## Validity obligations

The baseline is locked only after all of the following pass:

- every reachable state has legal continuation or an explicit terminal result;
- retry states have positive resolving mass or settle as no judgement;
- repeated information keys have one actor and one ordered action tuple;
- repeated information keys preserve the acting player's own prior
  information/action sequence;
- sampled and enumerated chance agree on reduced supports;
- returns are finite, zero-sum, and in `{-1,0,1}`;
- no policy lookup receives hidden truth.

Failure invalidates the model or implementation. It must not be repaired inside
the solver.

## Experimental models

Every alternative gets a new model ID, manifest, tables, and evaluation. Candidate
families are:

| Family | Change | Minimum evidence discipline |
|---|---|---|
| `historical-script-v1` | Deterministic or small-support upper outcomes reproduce depicted rounds. | Cite the round and distinguish replay from a general policy. |
| `softmax-upper-v1` | Legal exact hands receive probability proportional to `exp(beta*score(H,I))`. | Publish features, normalization, beta, and permitted upper information. |
| `failure-upper-v1` | Separate accidental failure from structural impossibility. | State cause observability and retry behavior. |
| `typed-upper-v1` | Draw persistent hidden upper skill types from a common prior. | Publish the prior and type persistence. |
| `private-root-v1` | Draw metal-card allocation and/or residual air at root. | Publish support, probabilities, and private observations. |

No alternative may reuse baseline regret tables. Cross-model evaluation is
allowed; cross-model training state is not.

## Model fingerprint

Every run records a canonical serialization and hash of:

- model ID and schema version;
- engine commit and dirty-tree digest;
- root inputs;
- upper policy identity and parameters;
- information-state schema;
- utility and retry rules;
- action ordering and numeric precision.

Two runs with different fingerprints solve different games. Results must be
reported under the full model ID, never as an unconditional solution to “Air
Poker.”

## Change protocol

1. Add or revise an evidence/gap entry.
2. Create a new model ID when behavior, information, chance, root, or utility changes.
3. Add transition and information tests before solver work.
4. Recompute the fingerprint.
5. Train and evaluate from empty tables.
6. Compare models only in a labeled cross-model report.
