# Golden fixtures for `ap`

Source: the Python Air Poker reference in `games/ap`, commit `cadd581`
("Add the robust public-state solver, engine self-play and the allocation loop").

Generator: `generate.py` in this directory. Run it from `games/ap`:

```sh
cd games/ap
PYTHONPATH=. uv run python -B <this directory>/generate.py            # every fixture
PYTHONPATH=. uv run python -B <this directory>/generate.py robust     # one fixture
```

| File | Python source | Content |
|---|---|---|
| `dp.json` | `src/kernel/dp.py` | `count` for targets 0..69 and `unrank` at four ranks per canonical target, on the full deck, the 19-card reduced deck and ten seeded decks |
| `sampler.json` | `src/kernel/sampler.py` | `stream` words, chained `bounded` draws, two `sample_sequence` count arrays (nonzero cells), four `sample_rounds` counts and twenty `sample_round_hands` results |
| `confidence.json` | `src/kernel/confidence.py` | `interval` over a grid of `(k, n, alpha)` up to `n = 3e8`, and five `boxes` at the solver's per-entry level |
| `kernel_round1.json` | `docs/results/kernel-round1.json` | the exact round-1 class counts of the 25 canonical pairs, copied with the recorded wall time |
| `robust.json` | `src/solver/robust.py` | the four sequences extending the prefix `(25,8),(26,44),(36,15)` sampled 30,000 times each (seed 9, batch = sequence index); 400 sampled values of the round-4 subtree `V_3`, 100 sampled strategy mixes, and 400 sampled values of one round-5 table `W_4`, at `delta = 1e-3` |
