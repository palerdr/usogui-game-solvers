import Base

# A failed check adds a dose of ST + 60. Both capacity limits must hold.
def survives(+st: U32, ttd: U32) -> Bool:
  (st < 240 : U32) && (st + ttd <= 240 : U32)

# TTD 0 contributes 240 profiles, then each successive row loses one.
def alive_id(st: U32, +ttd: U32, initial: Bool) -> U32:
  match initial:
    case True{}:
      st
    case False{}:
      +row = (ttd - 60 : U32)
      (240 + row * (363 - row) / 2 + st : U32)

def quotient(+st: U32, +ttd: U32, alive: Bool) -> U32:
  match alive:
    case False{}:
      (16711 + st : U32)
    case True{}:
      alive_id(st, ttd, U32.is_eq(ttd, 0))

def profile_id(+st: U32, +ttd: U32) -> U32:
  quotient(st, ttd, survives(st, ttd))

def success_child(grown: U32, ttd: U32, terminal: Bool) -> U32:
  match terminal:
    case True{}:
      17011
    case False{}:
      profile_id(grown, ttd)

def success(st: U32, ttd: U32, lag: U32) -> U32:
  +grown = (st + lag : U32)
  success_child(grown, ttd, (grown >= 300 : U32))

def failed_child(st: U32, ttd: U32, alive: Bool) -> U32:
  match alive:
    case False{}:
      17011
    case True{}:
      profile_id(0, (ttd + st + 60 : U32))

def failed(+st: U32, +ttd: U32) -> U32:
  failed_child(st, ttd, survives(st, ttd))

def revival_probability(st: U32, ttd: U32, alive: Bool) -> F64:
  match alive:
    case False{}:
      0.0
    case True{}:
      s = F64.from_nat(U32.to_nat(st))
      t = F64.from_nat(U32.to_nat(ttd))
      (0.95 * (1.0 - s / 240.0) * F64.pow(0.75, (t / 60.0 : F64)) : F64)

def revival(+st: U32, +ttd: U32) -> F64:
  revival_probability(st, ttd, survives(st, ttd))

# Child values come from the next Dropper's perspective.
def failed_payoff(child: F64, +p: F64) -> F64:
  (p * F64.neg(child) + (1.0 - p) : F64)
