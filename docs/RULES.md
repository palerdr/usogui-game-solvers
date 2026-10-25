# Frozen game rules

`src/stl/game.py` owns the live STL referee and action legality. DTH and the
leap solver implement these same physical rules. The documentary evidence is
indexed in [game-sources/EVIDENCE.md](game-sources/EVIDENCE.md); the complete
match ledger is [IN_DEPTH_SUMMARY.md](game-sources/IN_DEPTH_SUMMARY.md).

## Actions and simultaneous play

Actions are literal seconds 1 through 60. Action 1 means acting immediately;
action 0 and passing are illegal. A check succeeds when `check >= drop`.
Its squandered time is `ST = check - drop + 1`, including the first second.
A same-second check therefore adds one second, never zero.

Drop and check are private simultaneous commitments. Both actions become
public only after resolution. The server samples Hal's action inside submit;
the browser never receives an unrevealed current action.

In STL's leap window, only Baku as Dropper may additionally choose 61.
Checker remains capped at 60, so that drop always defeats the check. Both
players know the leap rule from initialization. DTH has no leap action.
STL policy arrays use literal indices: 0 is padding, 1..60 are ordinary
seconds, and 61 is the conditional Baku drop. DTH policies have 60 entries.

## Loads, death, and revival

Each player has vial contents `s` and cumulative time-to-death `t`, in seconds.
A successful check adds ST to the Checker's vial. Capacity is 300 seconds:
reaching it triggers an injection with physical dose capped at 300. A failed
check adds the fixed 60-second penalty, giving nominal dose `q = s + 60`.

- A current dose `q >= 300` has zero revival probability.
- `t + q > 300` is fatal. Equality remains eligible when `q < 300`.
- Revival uses prior TTD `t`. On survival, the vial resets and TTD becomes
  `t + q`; roles swap. On permanent death, the current Dropper wins.
- For integer seconds, eligibility is exactly `s <= 239 and s + t <= 240`.

The frozen probability is

$$
P_{\mathrm{rev}}(s,t)=
\begin{cases}
0.95(1-s/240)\,0.75^{t/60}, & s\le239\text{ and }s+t\le240,\\
0, & \text{otherwise}.
\end{cases}
$$

The capacity, penalty, eligibility boundaries, and qualitative effect of
prior damage come from the sources. The constants 0.95 and 0.75 and the
linear/geometric shape are declared modeling choices, not documentary odds.
There is no extra CPR-count factor, physicality multiplier, or referee gate.
The former referee floor cannot bind in the eligible region (`t <= 240`).

The probability falls with either input: `P(0,0)=0.95`,
`P(239,0)=0.95/240`, and `P(0,240)=0.95*0.75^4`. Equal total dose does not
imply equal odds: acute vial contents and prior damage play different roles.
For bucket width `B`, substituting `s=B*l` and `t=B*tau` gives exactly the
same surface, with capacity `C=300/B` and penalty `F=60/B`.

## STL clock and identities

The canonical opening is 8:12:00 AM (clock 720 measured from 8:00), round 1,
half 1, empty vials and TTD. Hal drops in half 1; Baku drops in half 2.
Normal half-rounds last 60 seconds. A half-round starting in the closed
interval `[3540,3600]` spans the leap second and lasts 61 seconds.

A survived injection adds its dose plus 120 seconds of procedure time.
Between halves, add 60 seconds. At round end, advance to the next whole-minute
boundary, treating clock 3600 as 8:59:60 and 3601 as 9:00:00. Thus an ordinary
no-death round advances four minutes. The solver's `child_clock` and referee
are checked against the same timing. [LEAP.md](LEAP.md) describes the derived
route and exact post-leap DTH boundary.

Hal's utility is +1 for a win and -1 for a loss; Baku's is its negative.
There is no canonical draw. Public history records revealed actions and
outcomes. Narrative memory interpretations do not change physical legality
or revival odds and do not define a counterfactual awareness model.

## Supported formulations

| Rung | Scope | Status and authority |
| --- | --- | --- |
| L0 | Pure DTH in 10- or 5-second buckets | Retired; solved research and its code are in `938fb25` |
| L1 | Pure DTH, seconds 1..60 | Complete 289,374,121-class table; `stl.dth` and `stl.dth_kernel`; [DTH.md](DTH.md) |
| L2 | Public STL clock and Baku's leap drop | Complete reachable value artifact through clock 720; `stl.leap` and its frozen kernel; [LEAP.md](LEAP.md) |

Each stage is a zero-sum simultaneous-action matrix game. Public state and
resolved history are observed; current actions are hidden until resolution.
No surviving code claims a solution of hidden-memory or psychological games.
Exact Hal uses L1; Translated Hal exploits a learned public-history forecast.
Neither browser provider uses the 82 GB L2 artifact or certifies full-STL
optimality. See [HAL.md](HAL.md).

Artifacts built with action 0, exclusive ST, different revival constants, or
a different leap mask are incompatible. Changes to rules must update the
evidence, docs, and tests together; solver gaps, tolerances,
and validation gates must not be weakened.
