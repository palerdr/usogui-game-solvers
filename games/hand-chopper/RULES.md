# Hand Chopper rules

**Status:** settled with the owner on 2026-11-16. No engine exists yet.
**Source:** [`hand-chopper-rules.jpg`](hand-chopper-rules.jpg), the rules page of *Usogui* chapter 391.

Hand Chopper is the manga's version of the hand game chopsticks. This file fixes the game that
the Python engine implements: the rules page as written, plus the finger cut that the referee
allowed in the match.

## Players and hands

Two players, `P0` and `P1`. `P0` moves first. Each player has two hands. A hand has two numbers:

```text
total   fingers the hand has            1..5, starts at 5
count   fingers the hand points out     1..total
```

A hand is **live** when `count < total`. A hand is **dead** when `count == total`: all its
fingers point out (rule 3). A dead hand stays dead. It cannot touch and it cannot be touched.

The game starts with four live hands, each with `count = 1` and `total = 5` (rule 1).

## Turn

The players alternate turns. A turn has two steps.

1. **Cut (optional).** The mover removes one finger from one of their own live hands. `total`
   drops by one and `count` does not change. A hand whose `count` now equals its `total` is
   dead. The mover may cut again, any number of times.
2. **Touch (mandatory).** The mover picks one of their own live hands and one of the opponent's
   live hands. The touched hand changes (rules 2 and 4):

```text
sum   = touched.count + touching.count
count = sum mod touched.total
count == 0  ->  the touched hand is dead
```

With `total = 5` this is the page's rule: a sum of exactly 5 kills the hand, and a larger sum
keeps the excess (3 touched by 3 gives 1). A player cannot pass and cannot touch their own hand.

## End of the game

A player loses at the moment both of their hands are dead (rule 5). The opponent's touch causes
this, or the player's own cut does. Returns to `(P0, P1)`:

```text
P1 has no live hand:   (+1, -1)
P0 has no live hand:   (-1, +1)
play never ends:       ( 0,  0)
```

A cut has no cost in the returns. The game is finite in cuts (20 fingers) and can repeat
positions through touches, so play that never ends is possible and counts as a draw.

The game is two-player, zero-sum, with perfect information and no chance moves.

## Decisions the owner settled

| Choice | Decision | Rejected |
| --- | --- | --- |
| Core rule set | The manga rules only | Real-life chopsticks as the core, or as switches in the core |
| Effect of a cut | `total` drops by one, `count` stays | The player picks a pointing finger, which also lowers `count` |
| Number of cuts | No limit | One cut per player; one cut per hand |
| Cutting your own finger | Legal | Illegal |

## Points the manga leaves open

The page and the match do not settle these points. The agent proposed the plainest reading of
each, and the owner confirmed all six on 2026-11-16.

- **A cut does not use the turn.** The mover cuts and then touches in the same turn. The
  reconstructed match line below needs this: with a cut that uses the turn, Baku loses.
- **A sum can wrap more than once.** A hand with `total < 5` can receive a sum of `2 * total`
  or more. The last position of the fingers decides, so the rule is `sum mod total`.
- **A dead hand takes no cut.** The cut would change nothing.
- **A cut can kill the mover's own hand.** A hand at `count = total - 1` dies from a cut. This
  follows from the cut rule with no special case.
- **No clock.** The match gave each player 5 minutes in total. The model has no time, so it
  uses the draw above where the match used the clock.
- **No wrong-count loss.** The page lists "pointing out the wrong number of fingers" as a way to
  lose. The engine applies the arithmetic, so a wrong count cannot occur.

## Evidence for the cut

The rules page does not mention a cut. The chapter pages that show it are not in this
repository. The account below comes from Japanese fan recaps of volume 36, which agree on these
points:

- Baku cut off his own little finger with a cigar cutter, on his turn, before his touch.
- The referee, Yakou, allowed it. The rules say "all the fingers" of an arm and name no number.
- The hand then had four fingers, and Baku won.

Recaps:
[readagain-comic.com](https://readagain-comic.com/%E4%BC%BD%E7%BE%85%E3%81%AE%E6%88%A6%E6%AD%BB%E3%82%92%E4%BB%A3%E5%84%9F%E3%81%AB%E5%9B%BD%E7%8E%8B%E3%81%AB%E4%B8%8A%E3%82%8A%E8%A9%B0%E3%82%81%E3%81%9F%E8%B2%98%E3%80%81%E3%81%9D%E3%81%97%E3%81%A6),
[note.com](https://note.com/tmknnm/n/n8664ae37324a),
[takehana-blog.com](https://takehana-blog.com/cccc/?p=2275).

One recap gives the position: each player has one live hand, Baku shows 2, Nerone shows 1, and
Baku is to move. With one live hand each, no player has a choice, so the line is forced. The
engine must reproduce these three lines (`B` = Baku's count, `N` = Nerone's count):

```text
No cut, Baku loses in 2 touches:
  B2 N1 -> B touches, N3 -> N touches, B dead (2 + 3 = 5)

Cut, then touch, Baku wins in 5 touches (Baku's total is 4):
  B2 N1 -> N3 -> B1 (5 mod 4) -> N4 -> B1 (5 mod 4) -> N dead (4 + 1 = 5)

Cut that uses the turn, Baku loses in 7 touches (Nerone moves first):
  B2 N1 -> B3 -> N4 -> B3 (7 mod 4) -> N2 (7 mod 5) -> B1 (5 mod 4) -> N3 -> B dead (1 + 3 = 4)
```

The position is a reconstruction from one recap. Check it against the chapter before you cite
it as the match record.

## Excluded variants

These belong to real-life chopsticks. Adding one defines a different game.

- **Transfers.** A player moves fingers between their own hands in place of a touch, or revives
  a dead hand that way.
- **Death at five or more.** A hand dies when the sum reaches or passes five, with no wrap.
- **Touching your own hand.**
- **Touching a dead hand.**

Also excluded: the clock, speed of play, tells, the stake, and the referee.
