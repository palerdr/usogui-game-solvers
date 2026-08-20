# Exact CFR+ unit-test game

`TinyAirPoker` is the smallest useful Air Poker fixture for exhaustive solver
tests. It has two lower players with two (or, by configuration, three) unique
metal cards each. A round keeps the production information order: Team Zero
selects privately, Team One selects without seeing that choice, both choices
become public, a complete correlated upper-hand outcome is sampled but hidden,
and the lower selected number acts first in a check/bet street.

The default upper support has only three complete hand-pair outcomes. It is
deliberately tiny so `CFRPlus` can enumerate every chance/action history.

```python
from src.exact import CFRPlus, TinyAirPoker

solver = CFRPlus(TinyAirPoker(), average_delay=0)
solver.train(2_000)
value = solver.evaluate_average_profile()
```

This package does not modify or attempt to enumerate the production five-card
engine. It is the exact-CFR+ correctness oracle for a small game only.
