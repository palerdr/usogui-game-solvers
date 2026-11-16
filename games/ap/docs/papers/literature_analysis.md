# Literature Analysis for a Reduced Air Poker Solver

**Scope:** a rigorous solver for the version of Air Poker in which the two lower players are the only strategic agents and the two upper constructors are represented by fixed stochastic policies.

**Companion specification:** `ENGINE.md`

**Recommended first solver:** exact tabular CFR+ as a reference implementation, followed by external-sampling MCCFR when the upper-hand chance branch becomes too large.

---

## 1. Executive conclusion

Once the upper players are replaced by fixed stochastic policies, the strategic game is a finite two-player, zero-sum extensive-form game with imperfect information and chance, provided that all of the following are fixed before a solve:

1. the prior over the lower players' private metal-card allocations;
2. the upper-player latent-type prior, if latent skill is used;
3. the upper construction policies and failure process;
4. the observation function for each lower player;
5. the terminal survival utility;
6. every timing and no-rank convention inherited from the engine.

That reduction makes CFR-family methods theoretically appropriate. The most important qualification is that a fixed hidden engine state is not by itself an imperfect-information game. A solver needs a probability model over the hidden initial states. If each lower player knows only their own metal cards, then the game tree must contain alternative deals that are grouped into information sets, together with a common prior over those deals. The canonical Baku/Lalo allocation can be one realized deal or one conditional evaluation point, but it cannot be the only root state if the opponent's cards are intended to be genuinely uncertain.

The five papers in this review divide into two layers:

| Layer | Paper | Function in the project |
|---|---|---|
| Lower-game equilibrium | Zinkevich et al. | Defines CFR, counterfactual regret, information-set decomposition, perfect-recall requirements, and equilibrium averaging. |
| Lower-game acceleration | Tammelin | Defines CFR+, regret-matching+, alternating updates, and linearly weighted averaging. |
| Chance-branch scaling | Lanctot et al. | Defines MCCFR, especially external sampling for sampling upper outcomes while enumerating the updating lower player's actions. |
| Upper behavioral policy | McKelvey and Palfrey | Supplies the logit quantal-choice foundation for a bounded-rational stochastic constructor. |
| Persistent upper uncertainty | Harsanyi | Supplies the root type draw and common-prior representation for persistent hidden skill parameters. |

The resulting architecture should be understood as:

```text
Harsanyi root chance over hidden types and private deals
    -> fixed upper stochastic construction kernel
    -> exact correlated hand-pair outcome
    -> two strategic lower players
    -> CFR+ or MCCFR equilibrium computation
```

The upper softmax model is a behavioral model inside chance. It is not automatically a quantal response equilibrium, because the upper players are no longer strategic agents in the reduced game and the score function is not necessarily their equilibrium continuation payoff.

---

## 2. Download links and bibliographic status

Among the five selected works, the CFR+ technical report is the one for which an official arXiv record was found. The remaining works are linked through their official proceedings, author repository, publisher, or an archival university mirror.

### Official arXiv PDF

1. **Oskari Tammelin, _Solving Large Imperfect Information Games Using CFR+_ (2014)**  
   [Direct arXiv PDF](https://arxiv.org/pdf/1407.5042)  
   [arXiv abstract page](https://arxiv.org/abs/1407.5042)

### Primary-source PDFs for the remaining core papers

2. **Martin Zinkevich, Michael Johanson, Michael Bowling, and Carmelo Piccione, _Regret Minimization in Games with Incomplete Information_ (NIPS 2007)**  
   [Official NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2007/file/08d98638c6fcd194a4b1e6992063e944-Paper.pdf)  
   [Official NeurIPS page](https://proceedings.neurips.cc/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html)

3. **Marc Lanctot, Kevin Waugh, Martin Zinkevich, and Michael Bowling, _Monte Carlo Sampling for Regret Minimization in Extensive Games_ (NIPS 2009)**  
   [Official NeurIPS PDF](https://papers.neurips.cc/paper_files/paper/2009/file/00411460f7c92d2124a67ea0f4cb5f85-Paper.pdf)  
   [Official NeurIPS page](https://proceedings.neurips.cc/paper/2009/hash/00411460f7c92d2124a67ea0f4cb5f85-Abstract.html)

4. **Richard D. McKelvey and Thomas R. Palfrey, _Quantal Response Equilibria for Extensive Form Games_ (1998)**  
   [Caltech author-repository page and downloadable working-paper PDF](https://authors.library.caltech.edu/records/prxgj-mg253)  
   [Published article DOI](https://doi.org/10.1023/A:1009905800005)

5. **John C. Harsanyi, _Games with Incomplete Information Played by “Bayesian” Players, Part I: The Basic Model_ (1967)**  
   [Publisher DOI and abstract](https://doi.org/10.1287/mnsc.14.3.159)  
   [University-hosted archival PDF](https://www2.cs.siu.edu/~hexmoor/classes/CS491-F10/Harasyani.pdf)

---

## 3. Formal statement of the reduced game

Let the two lower players be indexed by \(i\in\{0,1\}\). Let chance be denoted by \(c\). A finite extensive-form representation contains:

- a set of histories \(H\);
- terminal histories \(Z\subset H\);
- an actor function \(P(h)\in\{0,1,c\}\);
- legal actions \(A(h)\);
- lower information partitions \(\mathcal I_0,\mathcal I_1\);
- a fixed chance kernel \(f_c(a\mid h)\);
- zero-sum terminal utility \(u_0(z)=-u_1(z)\).

For this project, chance has several logically distinct jobs:

1. draw the lower private metal-card allocation;
2. draw persistent upper-player types, if used;
3. realize accidental upper failures;
4. realize the exact upper hand pair;
5. realize any explicitly modeled retry process;
6. resolve any genuinely stochastic physical rule retained by the engine.

These should remain distinguishable in state and logs even when several are sampled together for efficiency.

### 3.1 Initial lower-card uncertainty is mandatory

Suppose the canonical ten metal numbers are common knowledge but their five-five allocation is private. Then a natural root model is:

\[
(M_0,M_1)\sim \rho_M,
\qquad M_0\cap M_1=\varnothing,
\qquad M_0\cup M_1=M_{\text{all}}.
\]

Each player observes only \(M_i\). Histories with the same observed hand and same remembered public history belong to the same information set for that player.

If the solver instead initializes a single canonical deal and merely hides fields in an observation object, there are no alternative histories for the information set to contain. The implementation may look partially observable, but the equilibrium model is degenerate. A prior over alternative deals is therefore not an optional modeling flourish; it is part of the game definition.

Possible choices include:

- uniform over all five-five partitions of the ten known metal cards;
- a rule-derived distribution matching how the cards are canonically assigned;
- a deliberately narrow prior around the canonical deal;
- a custom experimental prior used for sensitivity analysis.

The prior must be recorded in solver output because a lower strategy is conditional on it.

### 3.2 Upper construction as a fixed chance kernel

For upper player \(i\), target \(n_i\), available deck \(D\), allowed upper observation \(I_i^U\), and latent type \(\theta_i\), define the legal construction set

\[
L(D,n_i)=\{H\subseteq D:\ |H|=5,\ \operatorname{rankSum}(H)=n_i\}.
\]

A complete upper event should preserve at least:

```python
UpperEvent(
    hand_0: int | None,
    hand_1: int | None,
    cause_0: UpperOutcomeCause,
    cause_1: UpperOutcomeCause,
    attempts: int,
)
```

The exact pair, not a collection of marginal summaries, determines:

- the poker winner;
- exact-card overlap and calamity;
- the union of removed cards;
- future legal construction sets;
- Witch Pain assignment;
- evidence available to the lower players.

If upper policies are conditionally independent, the pair probability can factor as

\[
P(H_0,H_1\mid s,\theta_0,\theta_1)
=
\pi_{\theta_0}(H_0\mid I_0^U(s))
\pi_{\theta_1}(H_1\mid I_1^U(s)).
\]

Both hands must nevertheless be sampled against the same pre-round deck. The implementation must not remove \(H_0\) before sampling \(H_1\), because that would make overlap impossible.

### 3.3 Information-set discipline

The full engine state contains hidden truth. The CFR key must contain only what the acting lower player observes and remembers.

A lower key may include:

- the player's own initial and remaining metal cards;
- the player's own previous metal choices;
- publicly revealed current and past targets;
- public betting history;
- public BIOS counts;
- the player's own exact hidden residual air, if retained;
- public round results and calamity signals;
- every prior observation needed for perfect recall.

It must not include:

- the opponent's unplayed metal cards;
- the exact hidden upper hand pair;
- the exact hidden remaining deck unless publicly revealed;
- the opponent's residual air;
- latent upper skill types;
- hidden failure causes;
- any future chance sample.

A valuable invariant is:

```python
if lower_observation(s1, i) == lower_observation(s2, i):
    assert information_state_key(s1, i) == information_state_key(s2, i)
```

The converse does not have to hold if a formally proven sufficient-state compression is used, but it should hold in the initial reference implementation.

---

# 4. Paper 1: Zinkevich et al. — CFR foundations

## Citation

Martin Zinkevich, Michael Johanson, Michael Bowling, and Carmelo Piccione. “Regret Minimization in Games with Incomplete Information.” Advances in Neural Information Processing Systems 20, 2007.

## Direct relevance

This is the foundational paper for the lower-player solver. It introduces counterfactual regret, proves that overall regret is bounded by the sum of positive immediate counterfactual regrets at information sets, and connects low regret in two-player zero-sum self-play to approximate Nash equilibrium.

The applicability is direct only if the reduced Air Poker game is:

- finite;
- two-player;
- zero-sum;
- represented in extensive form;
- equipped with fixed chance probabilities;
- equipped with perfect-recall lower information sets.

Those are engineering obligations, not incidental mathematical language.

## 4.1 The central decomposition

For lower player \(i\), information set \(I\), and action \(a\), CFR evaluates the value of forcing \(a\) at \(I\) while leaving the remaining strategy unchanged. The counterfactual reach weight excludes the updating player's own probability of reaching \(I\), but includes the opponent and chance.

Schematically:

\[
r_i^t(I,a)
=
\pi_{-i}^{\sigma^t}(I)
\left(v_i(\sigma^t_{I\to a},I)-v_i(\sigma^t,I)\right).
\]

The original result shows that controlling these local regrets controls overall regret. That is why the solver can maintain a table per information set instead of solving the normal form, whose strategy space would be exponentially larger.

### Air Poker interpretation

At a lower betting information set, the counterfactual weight must include:

- the probability of the opponent's private metal-card holding;
- the probability of persistent upper types;
- all prior stochastic upper hand pairs;
- accidental failures and retries;
- the opponent's lower strategy reach;
- all other chance events.

It excludes only the updating lower player's own reach contribution.

This is easy to get wrong when the upper model is implemented as an external simulator. If the chance probability is omitted from counterfactual reach, rare upper outcomes receive the same regret weight as common outcomes and the computed strategy solves the wrong game.

## 4.2 Perfect recall is load-bearing

CFR's standard guarantee assumes perfect recall. A lower player must never be forced by state compression to forget:

- which private metal cards they initially held;
- which metal card they chose in each prior round;
- which lower actions they previously took;
- which observations they received;
- any private air information that influenced earlier choices.

The current public state may be insufficient. Two histories can have the same current BIOS totals, pot, round, and remaining metal cards while differing in a player's remembered prior evidence about upper behavior. If those histories can lead to different beliefs, they cannot be merged casually.

A safe first implementation uses full immutable observation histories in the information key. Compression should come only after tests demonstrate strategic equivalence.

## 4.3 Average strategy, not only the current policy

CFR's equilibrium guarantee applies to an appropriately reach-weighted average strategy. For player \(i\):

\[
\bar\sigma_i^T(I,a)
=
\frac{\sum_{t=1}^{T}\pi_i^{\sigma^t}(I)\sigma_i^t(I,a)}
{\sum_{t=1}^{T}\pi_i^{\sigma^t}(I)}.
\]

A simple arithmetic mean over iteration-level action probabilities is not equivalent. Information sets that the player rarely reaches under their own strategy should not be weighted as if they were reached every iteration.

For the project, store separately:

```python
regret_sum[info_key][action]
strategy_sum[info_key][action]
```

Do not overwrite one with the other, and do not evaluate exploitability only on the current regret-matched strategy.

## 4.4 Utility definition

CFR will solve whatever terminal utility the engine supplies. It does not know the intended narrative objective.

These are distinct games:

```python
u = final_bios_0 - final_bios_1
```

and

```python
u = +1 if lower_0_survives_and_wins else -1
```

If the objective is faithful survival, use the engine's actual terminal winner. Intermediate BIOS changes are state transitions, not rewards. A margin-based utility may choose a line with larger expected BIOS advantage but lower survival probability.

Binary \(\{-1,+1\}\) utility also keeps the utility range bounded and makes exploitability easier to interpret.

## 4.5 What this paper forces you to implement

1. A root chance model for private metal-card uncertainty.
2. Lower-player information sets, not full-state policy keys.
3. Perfect recall.
4. Fixed chance probabilities during a solve.
5. Counterfactual reach that includes chance.
6. Reach-weighted average strategies.
7. Exact or tested best responses for exploitability.
8. A zero-sum terminal utility matching survival.

## 4.6 Failure modes to test

- **Full-state leakage:** sampled upper hands appear in the information key.
- **Single-deal degeneracy:** the opponent's private hand is hidden in code but no alternative root deals exist.
- **Broken recall:** two histories with different own prior actions share a key.
- **Chance omitted from reach:** upper outcomes are misweighted.
- **Wrong average:** unweighted iteration means are used.
- **Nonstationary chance:** upper parameters change during CFR iterations.
- **Wrong payoff:** the solver optimizes BIOS margin instead of survival.

---

# 5. Paper 2: Tammelin — CFR+

## Citation

Oskari Tammelin. “Solving Large Imperfect Information Games Using CFR+.” arXiv:1407.5042, 2014.

## Direct relevance

CFR+ is the strongest first reference solver for this project because the lower game has a short horizon and relatively modest strategic action sets, while chance is complicated. CFR+ retains the tabular transparency needed to validate the game before introducing sampling or neural approximation.

## 5.1 Regret-matching+

CFR+ replaces cumulative regret with a nonnegative cumulative quantity:

\[
R_{i,+}^{t+1}(I,a)
=
\max\left
\{R_{i,+}^{t}(I,a)+r_i^t(I,a),0\right\}.
\]

The next policy is proportional to these nonnegative regrets:

\[
\sigma_i^{t+1}(I,a)
=
\begin{cases}
\dfrac{R_{i,+}^{t+1}(I,a)}{\sum_b R_{i,+}^{t+1}(I,b)},
& \text{if the denominator is positive},\\[6pt]
\dfrac{1}{|A(I)|}, & \text{otherwise}.
\end{cases}
\]

The clipping is the defining local update, but a faithful CFR+ implementation is not merely “vanilla CFR with negative regrets set to zero.” Tammelin's report uses alternating player updates and linearly weighted strategy averaging, with an optional averaging delay.

## 5.2 Alternating updates

A practical iteration is:

```text
traverse and update lower player 0 while player 1 is fixed
traverse and update lower player 1 while player 0 is fixed
```

This should be implemented explicitly rather than silently mixing simultaneous and alternating formulas. On a small test game, compare both variants; they should approach the same value, although their trajectories differ.

## 5.3 Weighted averaging

Tammelin uses a weight sequence of the form

\[
w_t=\max(t-d,0),
\]

where \(d\) is an averaging delay. Later iterations receive greater weight.

For the reference solver:

- begin with \(d=0\);
- store the current policy and weighted average separately;
- report exploitability for the weighted average;
- add delayed averaging only after the zero-delay implementation is verified.

The iteration weight multiplies the normal reach-weighted strategy contribution. It does not replace reach weighting.

## 5.4 Why Air Poker is a good CFR+ reference target

The strategic lower action spaces are limited:

- choose one remaining metal card;
- check, call, fold, or choose a legal raise amount;
- possibly choose an air-management action in a later extension.

The difficult part is chance and hidden state, not an enormous action alphabet. That is ideal for a transparent tabular implementation.

A reduced fixture can be solved exactly first:

```text
2 or 3 rounds
small custom deck
small private metal-card distribution
uniform upper constructor
no failure
coarse or canonical raise sizes
binary survival utility
```

CFR+ on this fixture becomes the oracle against which later MCCFR and RL implementations are checked.

## 5.5 What not to assume

The 2014 report is short and heavily empirical. Treat CFR+ as an engineering acceleration built on CFR's game-theoretic foundation, not as permission to disregard the assumptions of the original CFR analysis.

In particular:

- CFR+ does not repair information leakage;
- CFR+ does not make an undefined private-card prior valid;
- CFR+ does not justify changing chance probabilities during training;
- CFR+ does not make a general-sum or four-agent team model into a two-player zero-sum game;
- CFR+ does not remove the need to evaluate an average strategy carefully.

## 5.6 Reference update skeleton

```python
def regret_matching_plus(regrets: list[float]) -> list[float]:
    total = sum(regrets)
    if total > 0.0:
        return [r / total for r in regrets]
    p = 1.0 / len(regrets)
    return [p] * len(regrets)


def update_regrets_plus(
    regrets: list[float],
    action_values: list[float],
    node_value: float,
    counterfactual_reach: float,
) -> None:
    for a in range(len(regrets)):
        instantaneous = counterfactual_reach * (
            action_values[a] - node_value
        )
        regrets[a] = max(0.0, regrets[a] + instantaneous)
```

This sketch omits traversal details, chance weighting, and strategy accumulation, but it fixes the local RM+ semantics.

## 5.7 Tests specific to CFR+

- regrets never become negative;
- an all-zero regret vector produces a uniform policy;
- alternating updates do not mutate the non-updating player's regrets;
- weighted average contributions use both iteration weight and the correct reach weight;
- exploitability falls on a hand-solvable reference game;
- the exact Air Poker fixture value is reproducible across runs.

---

# 6. Paper 3: Lanctot et al. — MCCFR

## Citation

Marc Lanctot, Kevin Waugh, Martin Zinkevich, and Michael Bowling. “Monte Carlo Sampling for Regret Minimization in Extensive Games.” Advances in Neural Information Processing Systems 22, 2009.

## Direct relevance

This paper addresses the principal scaling problem created by the upper constructor model. For targets \(n_0,n_1\) and deck \(D\), exact traversal can branch over

\[
|L(D,n_0)|\,|L(D,n_1)|
\]

hand pairs before accounting for failure, latent types, retries, and future deck consequences.

MCCFR replaces complete traversal with unbiased or appropriately weighted sampled regret estimates.

## 6.1 Recommended variant: external sampling

When updating lower player \(i\), external-sampling MCCFR:

- enumerates all actions of player \(i\);
- samples the other lower player's actions;
- samples chance outcomes;
- updates regret only for player \(i\) on that pass.

This fits Air Poker because the lower branching factor is manageable, while the upper chance branch is potentially very large.

A conceptual traversal is:

```python
def traverse(state, updating_player):
    if terminal(state):
        return utility(state, updating_player)

    if state.actor is CHANCE:
        outcome = sample_from_true_chance_kernel(state)
        return traverse(apply_chance(state, outcome), updating_player)

    if state.actor == updating_player:
        values = []
        for action in legal_actions(state):
            values.append(traverse(apply(state, action), updating_player))
        update_counterfactual_regret(...)
        return policy_weighted_sum(values)

    action = sample(opponent_policy(state))
    return traverse(apply(state, action), updating_player)
```

The actual implementation must carry the required reach and sampling probabilities.

## 6.2 Sample the joint construction event

The chance sample should be a single coherent object:

\[
\omega_r=(\theta_0,\theta_1,H_0,H_1,F_0,F_1,R_r),
\]

where \(R_r\) contains retry information if needed. Persistent types are normally sampled at the root and referenced later rather than resampled inside every round.

This is wrong:

```python
winner = sample(winner_probability)
calamity = sample(calamity_probability)
removed = sample(deck_transition_probability)
```

The three variables are correlated consequences of the same hand pair.

This is correct:

```python
hand_0, cause_0 = sample_upper_0(pre_round_deck, target_0)
hand_1, cause_1 = sample_upper_1(pre_round_deck, target_1)

winner = compare(hand_0, hand_1)
calamity = bool(hand_0 & hand_1)
removed = hand_0 | hand_1
```

Both policies receive the same `pre_round_deck`.

## 6.3 Chance sampling and action alternatives

The location of the chance node matters.

If upper construction is committed before lower betting, then one sampled upper pair must be retained throughout all later lower betting alternatives in that sampled history. Resampling a different upper pair for every candidate bet would place chance after the bet and solve a different game.

At an earlier lower metal-choice node, different target actions legitimately lead to different upper chance distributions. Each action branch may therefore encounter its own chance node after the chosen target is applied.

The engine's causal order is the authority:

```text
private deal
-> metal choices
-> target reveal
-> upper construction chance
-> lower betting
-> settlement
```

A chance event may be moved earlier only if it is conditionally independent of every intervening strategic action and remains hidden in exactly the same way.

## 6.4 Actual-policy sampling versus proposal sampling

The easiest correct implementation samples chance from the actual modeled kernel \(K\). If a more convenient proposal \(q\) is used, the traversal needs importance correction based on \(K(\omega\mid s)/q(\omega\mid s)\).

This matters for softmax constructors. Sampling uniformly from legal hands and then pretending the draw came from the bounded-rational policy is biased unless weights correct the difference.

Before using a proposal distribution, verify:

- every positive-probability target outcome under \(K\) has positive probability under \(q\);
- the target and proposal probabilities can be computed accurately;
- importance ratios are not so extreme that variance becomes unusable.

## 6.5 Outcome sampling versus external sampling

Outcome sampling follows one terminal trajectory and can support online learning with samples of an opponent. It has lower per-iteration work but typically higher variance and requires an exploration policy with sufficient support.

External sampling is preferable initially because:

- all updating-player actions are evaluated each visit;
- lower action spaces are modest;
- the expensive upper chance branch is sampled;
- it is easier to compare against exact CFR+ on reduced fixtures.

Outcome sampling becomes attractive only if lower action branching or memory also becomes prohibitive.

## 6.6 Persistent latent types

A sampled upper type must remain fixed for the match:

```python
root_type_0, root_type_1 = sample_joint_type_prior()
```

Later rounds use those same types, modified only by explicit state variables such as Witch Pain.

This is usually wrong:

```python
for round in rounds:
    beta = sample_beta_again()
```

Per-round resampling destroys the cross-round consistency that lets lower players learn from prior upper outcomes.

## 6.7 No-rank and retries

A double no-rank is not one flat chance outcome if the retry policy receives new information. It is a sequential chance process:

```text
attempt 1 under initial upper observations
-> both no-rank
-> observations update, possibly revealing opponent target
-> attempt 2 under retry observations
-> ...
```

The stochastic kernel must preserve that causal update. If retries are truncated for computation, the truncation rule is part of the game and must be reported.

## 6.8 MCCFR validation tests

1. On a small fixture, sampled estimates converge to exact CFR+ values.
2. A sampled hand pair is reused throughout downstream betting branches.
3. Both upper hands are sampled from the same pre-round deck.
4. Persistent types are not resampled mid-match.
5. Every target chance outcome has support under the sampling policy.
6. Importance ratios equal one when sampling directly from the modeled kernel.
7. Repeated runs produce confidence intervals containing the exact fixture value.

---

# 7. Paper 4: McKelvey and Palfrey — logit quantal choice

## Citation

Richard D. McKelvey and Thomas R. Palfrey. “Quantal Response Equilibria for Extensive Form Games.” Experimental Economics 1, 9–41, 1998.

## Direct relevance

The paper supplies the most defensible theoretical lineage for a stochastic upper policy of the form

\[
P(H\mid I,\theta)
\propto
\exp\bigl(\beta U(H,I)\bigr).
\]

It models actions as noisy maximization and, under a type-I extreme-value disturbance, obtains multinomial-logit choice probabilities.

## 7.1 Correct terminology

Your constructor can be called a:

> fixed logit quantal-choice policy inspired by logit-AQRE.

It should not automatically be called an AQRE.

In McKelvey and Palfrey, AQRE is a fixed point: each decision agent's logit choice probabilities are generated from expected continuation payoffs under the strategy profile that those same probabilities induce. In the reduced Air Poker game:

- the upper players are chance, not equilibrium players;
- \(U(H,I)\) is a designed score;
- the lower equilibrium does not require the upper policy to be a best response;
- the upper policies may be fitted, stipulated, or stress-tested.

The logit map is applicable; the equilibrium label is usually not.

## 7.2 Baseline policy

For nonempty \(L(D,n)\), a clean mixture is:

\[
P(\bot_{\mathrm{acc}}\mid I,\theta)=p_{\mathrm{fail}}(I,\theta),
\]

\[
P(H\mid I,\theta)
=
\left(1-p_{\mathrm{fail}}(I,\theta)\right)
\frac{\exp\left(\beta U(H,I)\right)}
{\sum_{G\in L(D,n)}\exp\left(\beta U(G,I)\right)}.
\]

If \(L(D,n)=\varnothing\), structural no-rank should occur with probability one. That is different from accidental failure.

A deliberate no-rank can be introduced later as either:

- a separate strategic-like outcome with its own utility; or
- a mixture component with a policy conditioned on communicated information.

Do not merge structural impossibility, accidental failure, and deliberate invalid play into one unlabeled `NoRank` probability. Their causes imply different updates, evidence, and counterfactual experiments.

## 7.3 Failure model

A logistic failure probability is coherent:

\[
p_{\mathrm{fail}}
=
\sigma\left(
\alpha_0
+\alpha_1 W
+\alpha_2 C(D,n)
-\alpha_3 T
\right),
\]

where:

- \(W\) is Witch Pain or impairment state;
- \(C(D,n)\) is construction difficulty;
- \(T\) is effective time available.

Difficulty should be defined operationally. Candidate features include:

- \(-\log(1+|L(D,n)|)\), so sparse legal sets are harder;
- search depth or node expansions required by the constructor algorithm;
- entropy of legal ranks or suits;
- evaluator complexity;
- distance from a previously cached construction.

If the engine is timeless, `timeAvailable` is a fixed rule parameter rather than a lower strategic action. If later lower delay can affect the upper clock, the chance node must be placed after that delay action.

## 7.4 Utility features and information legality

A starting score is:

\[
U(H,I)
=
w_s S(H)
+w_f F(D\setminus H,I)
+w_d Q(H,I).
\]

### Hand strength \(S(H)\)

Use a normalized quantity. A current-choice-set percentile is easy to interpret:

\[
S(H)
=
\frac{|\{G\in L(D,n):G\text{ ranks below }H\}|}
{\max(1,|L(D,n)|-1)}.
\]

It lies in \([0,1]\) and avoids arbitrary dependence on the raw evaluator's integer scale.

### Future flexibility \(F(D\setminus H,I)\)

The upper constructor may not know the ally's future targets unless they were communicated. Therefore flexibility should be an expectation under an allowed target belief:

\[
F(D\setminus H,I)
=
\mathbb E_{N\sim q_i(\cdot\mid I)}
\left[
\log\bigl(1+|L(D\setminus H,N)|\bigr)
\right].
\]

If the upper player has no legal basis for a future-target belief, use an explicitly stated generic prior or set the feature weight to zero.

### Expected denial \(Q(H,I)\)

A same-round opponent constructs against the pre-round deck, so choosing \(H\) does not remove cards from their current legal set. Denial is therefore mainly a future-round effect.

A legal formulation is:

\[
Q(H,I)
=
\mathbb E_{N_{-i}\sim q_{-i}(\cdot\mid I)}
\left[
\operatorname{lossOfFutureOptions}(H,N_{-i})
\right].
\]

It must not use the actual hidden opponent target before lock.

## 7.5 Score-scale identification

Only the product of \(\beta\) and utility scale determines the softmax. Multiplying every utility by ten and dividing \(\beta\) by ten leaves the policy unchanged.

Therefore a sensitivity grid such as \(\beta\in\{0,1,3\}\) is meaningful only after normalizing features and fixing weight conventions.

Recommended practice:

- normalize each feature to a documented range;
- fix one overall utility scale;
- interpret \(\beta\) relative to that fixed scale;
- record weights and normalization in every experiment manifest.

For numerical stability:

```python
max_score = max(scores)
weights = [math.exp(beta * (s - max_score)) for s in scores]
```

## 7.6 Choice-set multiplicity and the IIA issue

A softmax over exact five-card masks assigns probability to each exact hand. If many suit variants have the same strategic meaning and score, their category receives more total probability because it has more representatives.

That can be interpreted in two ways:

1. **Desirable maximum entropy over exact hands.** Every exact legal construction is a distinct cognitive option.
2. **Unwanted duplication effect.** The constructor first thinks in hand categories or rank patterns, so multiple equivalent suit realizations should not multiply category probability.

This decision is important in Air Poker because exact suits determine overlap and calamity. Three defensible models are:

- exact-hand softmax;
- hierarchical softmax: choose a structural hand class, then an exact realization;
- sequential construction model: choose cards one at a time.

The uniform baseline in the user's proposal is uniform over exact legal masks. Keep that interpretation explicit.

## 7.7 Additional score terms worth considering later

Without changing the initial baseline, later models may add:

- expected overlap or calamity value under permitted beliefs;
- probability the hand wins against an allowed opponent-hand distribution;
- cognitive search cost;
- robustness to accidental card misremembering;
- value of preserving specific suit structures;
- deliberate no-rank utility after communication.

Every added feature must be computable from the upper observation, not the omniscient engine state.

## 7.8 Tests for the constructor model

- \(\beta=0\) produces uniform probability over exact legal hands after conditioning on no failure;
- increasing \(\beta\) concentrates probability on maximum-score hands;
- probabilities sum to one;
- impossible target sets yield structural no-rank with probability one;
- two full states with identical upper observations induce identical upper distributions;
- actual opponent target changes do not alter the pre-lock policy unless they alter permitted observations;
- feature normalization is deterministic;
- repeated suit-equivalent hands receive the intended amount of aggregate mass.

---

# 8. Paper 5: Harsanyi — persistent hidden types

## Citation

John C. Harsanyi. “Games with Incomplete Information Played by ‘Bayesian’ Players, Part I: The Basic Model.” Management Science 14(3), 159–182, 1967.

## Direct relevance

Harsanyi's transformation gives the rigorous form of the proposal to draw an upper player's latent skill once at match start. Nature selects a hidden type from a commonly known prior; players observe only the information assigned to them; later behavior reveals evidence about the hidden type.

## 8.1 Type definition

A useful upper type is:

```python
@dataclass(frozen=True, slots=True)
class UpperType:
    beta: float
    strength_weight: float
    flexibility_weight: float
    denial_weight: float
    base_failure_logit: float
    witch_pain_coefficient: float
    difficulty_coefficient: float
    time_coefficient: float
```

The joint root draw is:

\[
(\theta_0,\theta_1)\sim\rho_\Theta.
\]

The prior need not factor:

\[
\rho_\Theta(\theta_0,\theta_1)
\neq
\rho_0(\theta_0)\rho_1(\theta_1).
\]

A correlated prior may represent common training, shared environment, or selection effects. Independence is a modeling choice, not a theorem.

## 8.2 Why draw once rather than every round

Persistent types create cross-round statistical structure:

- a strength-seeking constructor remains strength-seeking;
- a high-failure constructor is more likely to fail again;
- prior outcomes provide evidence about later construction behavior;
- Witch Pain modifies a baseline type instead of replacing it;
- lower players can rationally update beliefs.

Per-round parameter resampling instead models independent shocks. That may be useful as an additional noise component, but it should not replace persistent skill if cross-round learning is intended.

## 8.3 Bayesian updating is implicit in the exact game tree

In an exact CFR tree, a lower player does not need a manually stored posterior over \(\theta\). The hidden type histories, observation histories, and reach probabilities already induce the correct posterior at each information set.

Conceptually:

\[
P(\theta\mid h_i)
\propto
\rho_\Theta(\theta)
P(h_i\mid\theta,\sigma),
\]

but CFR can sum over the compatible histories directly.

An explicit posterior becomes useful when:

- compressing the information state;
- training an RL policy;
- using a recurrent or belief-state model;
- reporting human-readable diagnostics.

If a posterior is included in an observation vector, it must be derived from public and private observations available to that player. It cannot be computed using hidden engine truth and then handed to the agent.

## 8.4 Common prior is an assumption

The Harsanyi model uses a common basic probability distribution. For this project, that means both lower players know the same model family and prior over upper types, even though they do not know the realized types.

This is a strong but useful baseline. It should be stress-tested by solving under alternative priors. A strategy that changes dramatically under small prior changes is model-sensitive even if its within-model exploitability is tiny.

## 8.5 Prior mixture versus robustness

A root prior solves an expected-value problem:

\[
\max_\sigma
\mathbb E_{\theta\sim\rho}
[V_\theta(\sigma)].
\]

It does not solve:

\[
\max_\sigma
\min_{\theta\in\Theta}
V_\theta(\sigma).
\]

The latter is a robust or adversarial model. Do not call a Bayes-optimal strategy “robust” merely because the prior has several types.

The user's proposed sensitivity analysis is therefore appropriate:

1. solve separately under each fixed parameter model;
2. solve under selected latent mixtures;
3. evaluate every strategy under every alternative model;
4. report mean, worst-case, and exploitability separately.

## 8.6 Root chance ordering

A clean root order is:

```text
chance draws private metal-card deal
chance draws persistent upper types
each lower player observes own metal cards
play begins
```

The two chance draws may be sampled together if independent of later play, but they should remain separately logged.

## 8.7 Tests for the type model

- sampled type remains fixed across the match;
- lower observations never contain the type;
- histories generated by different types but identical observations share the same lower information set;
- public evidence changes the implicit posterior through reach probabilities;
- a degenerate prior reproduces the corresponding fixed upper model;
- an independent joint prior equals the product distribution numerically;
- correlated priors are supported without changing solver logic.

---

# 9. Integrated upper chance model

A defensible first implementation separates structural impossibility, accidental failure, deliberate no-rank, and legal hand choice.

## 9.1 Outcome causes

```python
class UpperOutcomeCause(Enum):
    LEGAL_HAND = auto()
    STRUCTURAL_NO_RANK = auto()
    ACCIDENTAL_FAILURE = auto()
    DELIBERATE_NO_RANK = auto()
```

Keeping causes distinct allows later evidence models. A public no-rank result may be observed without its cause being revealed.

## 9.2 Distribution

For a nonempty legal set and no deliberate no-rank:

\[
\pi_\theta(\bot_{\mathrm{acc}}\mid I)
=p_{\mathrm{fail}}(I,\theta),
\]

\[
\pi_\theta(H\mid I)
=(1-p_{\mathrm{fail}})
\operatorname{Softmax}_{H\in L(D,n)}(\beta U(H,I)).
\]

With deliberate no-rank probability \(p_{\mathrm{del}}\):

\[
P(\bot_{\mathrm{del}})
=(1-p_{\mathrm{fail}})p_{\mathrm{del}},
\]

\[
P(H)
=(1-p_{\mathrm{fail}})(1-p_{\mathrm{del}})
\operatorname{Softmax}(\beta U(H,I)).
\]

This mixture keeps accidental and intentional behavior identifiable in simulation even if they are observationally aliased to the lower players.

## 9.3 Joint pair kernel

If conditionally independent:

\[
K_\theta(e_0,e_1\mid s)
=
\pi_{\theta_0}(e_0\mid I_0^U(s))
\pi_{\theta_1}(e_1\mid I_1^U(s)).
\]

Conditional independence does not imply independent consequences. The pair is passed intact to settlement.

If the upper types are correlated, independence conditional on types can still hold. If there is shared randomness or communication, implement a direct joint kernel instead.

## 9.4 Transactional deck update

```python
pre_round_deck = state.available_deck_mask

event_0 = sample_or_traverse(policy_0(pre_round_deck, target_0))
event_1 = sample_or_traverse(policy_1(pre_round_deck, target_1))

mask_0 = event_0.hand_mask if event_0.is_legal else 0
mask_1 = event_1.hand_mask if event_1.is_legal else 0

next_deck = pre_round_deck & ~(mask_0 | mask_1)
```

Never mutate the deck before both distributions are evaluated against the same state.

## 9.5 Retry kernel

```python
while event_0.is_no_rank and event_1.is_no_rank:
    upper_obs_0 = reveal_retry_information(...)
    upper_obs_1 = reveal_retry_information(...)
    event_0 = sample_or_traverse(policy_0(upper_obs_0))
    event_1 = sample_or_traverse(policy_1(upper_obs_1))
```

A finite retry cap is a solver abstraction and should produce an explicit terminal or fallback event.

---

# 10. Suggested software boundary

The equilibrium solver should not know how upper probabilities are produced. It should see a standard chance-node interface.

```python
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class WeightedOutcome(Generic[T]):
    outcome: T
    probability: float


class UpperChanceModel(Protocol):
    def root_type_outcomes(
        self,
    ) -> tuple[WeightedOutcome[UpperTypePair], ...]: ...

    def construction_outcomes(
        self,
        observation_0: UpperObservation,
        observation_1: UpperObservation,
        type_0: UpperType,
        type_1: UpperType,
    ) -> Iterable[WeightedOutcome[UpperJointEvent]]: ...

    def sample_construction(
        self,
        rng: random.Random,
        observation_0: UpperObservation,
        observation_1: UpperObservation,
        type_0: UpperType,
        type_1: UpperType,
    ) -> tuple[UpperJointEvent, float]:
        """Return the event and its true modeled probability."""
```

The exact CFR+ solver uses `construction_outcomes`. MCCFR uses `sample_construction`. Both should rely on the same underlying scoring and probability code.

This dual interface is important. It lets reduced tests compare exact expectation against Monte Carlo estimates without maintaining two semantic implementations.

## 10.1 Recommended caches

```python
legal_hands_cache[(deck_mask, target)]
hand_strength_cache[hand_mask]
rank_sum_cache[hand_mask]
feature_cache[(upper_observation_key, hand_mask, type_id)]
normalizer_cache[(upper_observation_key, type_id)]
```

Be cautious with caching by full game state. The upper policy should depend only on `UpperObservation`; cache keys should make information restrictions visible.

## 10.2 Probability audit

Every chance state should support an audit mode:

```python
outcomes = tuple(model.construction_outcomes(...))
assert all(o.probability >= 0.0 for o in outcomes)
assert math.isclose(
    sum(o.probability for o in outcomes),
    1.0,
    rel_tol=1e-12,
    abs_tol=1e-12,
)
```

For sampled mode, log:

- modeled probability;
- proposal probability, if different;
- importance ratio;
- latent type;
- exact hand pair;
- resulting deck union;
- calamity indicator.

These logs are indispensable for debugging rare correlated outcomes.

---

# 11. Solver development sequence

## Stage 0: game-definition audit

Before CFR:

1. specify the lower private-card prior;
2. specify what lower players observe after each upper result;
3. specify terminal survival utility;
4. specify no-rank and retry behavior;
5. freeze timing assumptions;
6. verify all engine invariants.

## Stage 1: exact uniform upper constructor

Use:

\[
\beta=0,
\qquad p_{\mathrm{fail}}=0,
\qquad p_{\mathrm{del}}=0.
\]

Traverse every legal upper pair on a reduced fixture. This is the maximum-entropy exact-hand baseline.

## Stage 2: vanilla CFR

Implement vanilla CFR before CFR+ if possible. It is slower, but it is conceptually useful as a second reference. Compare game values and exploitability with CFR+.

## Stage 3: exact CFR+

Use:

- alternating player updates;
- regret-matching+;
- zero averaging delay initially;
- linearly weighted average strategy;
- exact chance traversal.

## Stage 4: external-sampling MCCFR

Sample:

- private deal when appropriate;
- root upper types;
- upper joint construction events;
- non-updating lower actions.

Enumerate the updating lower player's actions.

## Stage 5: bounded-rational upper models

A useful grid is:

\[
\beta\in\{0,1,3,10\},
\qquad
p_{\mathrm{fail}}\in\{0,0.01,0.05\}.
\]

Because \(\beta\) depends on utility scaling, publish the feature normalization with the grid.

## Stage 6: persistent type mixtures

Example discrete types:

```text
uniform
mildly strength-seeking
strong and low-failure
flexibility-preserving
impaired after Witch Pain
```

Start with a small discrete support. Ten observed submissions cannot identify a high-dimensional continuous type distribution reliably.

## Stage 7: cross-model evaluation

For models \(\phi_i\) and strategies \(\sigma_i^*\), compute

\[
M_{ij}=V_{\phi_j}(\sigma_i^*).
\]

Also compute exploitability of \(\sigma_i^*\) under \(\phi_j\). Value alone can hide that both players would change substantially under the alternate model.

---

# 12. Evaluation and reporting

## 12.1 Within-model metrics

- average-strategy exploitability;
- estimated game value;
- best-response values for both lower players;
- number of information sets reached;
- regret norm or positive regret mass;
- wall-clock time and nodes touched;
- chance samples per iteration;
- confidence intervals for sampled estimates.

## 12.2 Cross-model metrics

- value matrix \(M_{ij}\);
- worst-case value across the model grid;
- mean and variance across models;
- exploitability under each alternate model;
- reach-weighted policy distance;
- action-frequency changes at key information sets;
- probability of fold, showdown, calamity, no-rank, and each terminal cause.

## 12.3 Strategy stability

A lower strategy is more defensible when:

- its value remains acceptable across plausible upper models;
- its key action probabilities do not swing sharply with small parameter changes;
- exploitability remains low under the intended model;
- cross-model best responses do not find catastrophic weaknesses;
- conclusions are not driven by a tiny number of rare, high-weight samples.

## 12.4 Separate model uncertainty from solver error

Report two different uncertainties:

1. **solver approximation:** exploitability, regret, Monte Carlo confidence;
2. **behavioral model uncertainty:** variation across \(\beta\), feature weights, failure rates, and priors.

A numerically converged equilibrium can still be behaviorally fragile.

---

# 13. Critical modeling decisions that should be written into the contract

## 13.1 Exact-hand uniformity versus category uniformity

“Uniform legal construction” must say whether it is uniform over:

- exact rank-and-suit hand masks;
- poker hand categories;
- rank multisets;
- outputs of a particular search procedure.

The proposed definition \(1/|L(D,n)|\) is uniform over exact masks.

## 13.2 Lower private-card prior

No equilibrium exists without a common model of the opponent's hidden private cards. Record the prior and whether the canonical match is:

- a sampled realization;
- a conditional evaluation;
- or the entire game by common knowledge.

## 13.3 Public upper evidence

Specify whether lower players observe:

- only round winner;
- poker category;
- exact upper cards;
- overlap count;
- calamity only;
- no-rank occurrence;
- failure cause.

These choices alter information sets and posterior learning.

## 13.4 Chance timing

Specify when upper construction locks relative to lower betting and timing actions. Moving chance earlier or later changes the game unless conditional independence proves equivalence.

## 13.5 Persistent versus transient impairment

Witch Pain can affect:

- persistent type parameters;
- a public or hidden state variable;
- only the next construction;
- all remaining constructions.

Choose one explicit transition model.

## 13.6 Deliberate no-rank

Leave it at zero until communication is modeled. Once enabled, specify:

- what signal was observed;
- which future targets are believed;
- whether no-rank consumes cards;
- how the upper utility values team continuation.

---

# 14. Recommended reading order

## First pass

1. **Zinkevich et al.** — read the extensive-game definition, perfect recall, counterfactual regret, regret matching, and the regret-to-equilibrium connection.
2. **Tammelin** — read the complete six-page report and Algorithm 1.
3. **Lanctot et al.** — focus on the MCCFR framework, outcome sampling, external sampling, and the sampling-probability conditions.
4. **McKelvey and Palfrey** — focus on the AQRE definition and logit specialization; separate the response function from the equilibrium fixed-point concept.
5. **Harsanyi** — focus on the basic probability distribution, types, the prior lottery, and Bayes-equivalent representation.

## Implementation pass

Return to the papers in this order:

1. Zinkevich for reach probabilities and average strategy;
2. Tammelin for RM+ and weighted averaging;
3. Lanctot for sampled traversal and importance weighting;
4. McKelvey/Palfrey for upper policy calibration;
5. Harsanyi for persistent type priors and observation structure.

---

# 15. What each paper contributes—and what it does not

| Paper | Gives you | Does not give you |
|---|---|---|
| Zinkevich et al. | CFR theory, information-set regret decomposition, perfect-recall conditions, average-strategy equilibrium logic. | A behavioral upper model, a private-card prior, or an Air Poker utility definition. |
| Tammelin | RM+, alternating CFR+, weighted averaging, practical acceleration. | Permission to violate CFR assumptions or a sampled-chance derivation. |
| Lanctot et al. | Unbiased/sample-based CFR framework, external and outcome sampling. | A license to sample winner/calamity/deck independently or to ignore proposal probabilities. |
| McKelvey/Palfrey | Logit noisy-choice foundation and extensive-form quantal response concepts. | Evidence that your hand score is the true upper utility or that the reduced model is an AQRE. |
| Harsanyi | Root hidden types, common priors, persistent uncertainty, implicit Bayesian updating through the game tree. | Robust worst-case optimization or an empirically identified skill prior. |

---

# 16. Final architecture recommendation

Use the following separation:

```text
poker_math.py
    Exact bitwise card representation and hand evaluation

ENGINE.md / game.py
    Faithful deterministic game mechanics and observations

upper_model.py
    Legal-hand enumeration, feature computation, failure model,
    latent types, exact and sampled chance distributions

cfr.py
    Vanilla exact reference solver

cfr_plus.py
    Exact alternating CFR+ reference solver

mccfr.py
    External-sampling production solver

best_response.py
    Exact or independently validated exploitability computation

experiments.py
    Parameter grids, cross-model evaluation, manifests, reports
```

The most important rule is that `upper_model.py` receives `UpperObservation`, not `GameState`. The solver may know the full state to traverse chance, but the modeled upper policy must be a function only of information the upper player is allowed to possess.

The most defensible first scientific result is not a single claimed equilibrium under one speculative constructor. It is a family of lower equilibria across explicit upper models, accompanied by a cross-model value and exploitability matrix. A strategy that survives that analysis is far stronger evidence than a strategy optimized against one unidentifiable behavioral specification.
