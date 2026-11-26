import Base

# Strict comparisons keep the reference's first operands on ties.
def min(+a: F64, +b: F64) -> F64:
  Bool.pick(F64, F64.is_lt(b, a), b, a)

def max(+a: F64, +b: F64) -> F64:
  Bool.pick(F64, F64.is_gt(b, a), b, a)

def finite(+a: F64) -> Bool:
  F64.is_eq((a - a : F64), 0.0)

def gap_ok(lo: F64, hi: F64) -> Bool:
  F64.is_le((hi - lo : F64), (1.0 / 1000000.0 : F64))

type Scan is Type:
  Extrema{values: Array<F64>, lower: F64, upper: F64}

def scan(n: Nat, +i: U32, next: Array<F64> & F64, lo: F64, hi: F64) -> Scan:
  match n:
    case 0n:
      (s, +x) = next
      Extrema{s, min(lo, x), max(hi, x)}
    case 1n+rest:
      (s, +x) = next
      +j = (i + 1 : U32)
      scan(rest, j, Array.get(F64, s, j), min(lo, x), max(hi, x))

type Payoffs is Type:
  Payoff{values: Array<F64>, diffs: Array<F64>}

def differences(n: Nat, +i: U32, next: Array<F64> & F64, prev: F64, diffs: Array<F64>) -> Payoffs:
  match n:
    case 0n:
      (s, x) = next
      Payoff{s, Array.set(F64, diffs, i, (x - prev : F64))}
    case 1n+rest:
      (s, +x) = next
      differences(rest, (i + 1 : U32), Array.get(F64, s, (i + 2 : U32)), x,
        Array.set(F64, diffs, i, (x - prev : F64)))

type Product is Type:
  Dot{first: Array<F64>, second: Array<F64>, sum: F64}

# The second index descends for the recurrence and ascends for M q.
def dot(n: Nat, +m: U32, +j: U32, +stride: U32, first: Array<F64> & F64, second: Array<F64> & F64, acc: F64) -> Product:
  match n:
    case 0n:
      (a, x) = first
      (b, y) = second
      Dot{a, b, (acc + x * y : F64)}
    case 1n+rest:
      (a, x) = first
      (b, y) = second
      +m1 = (m + 1 : U32)
      +j1 = (j + stride : U32)
      dot(rest, m1, j1, stride, Array.get(F64, a, m1), Array.get(F64, b, j1), (acc + x * y : F64))

def recur_start(n: Nat, +j: U32, ds: Array<F64>, r: Array<F64>) -> Product:
  dot(n, 0, j, 4294967295, Array.get(F64, ds, 0), Array.get(F64, r, j), 0.0)

def recurrence(n: Nat, +k: U32, +d0: F64, product: Product) -> Array<F64>:
  match n:
    case 0n:
      Dot{_, r, acc} = product
      Array.set(F64, r, k, (F64.neg(acc) / d0 : F64))
    case 1n+rest:
      Dot{ds, r, acc} = product
      recurrence(rest, (k + 1 : U32), d0,
        recur_start(U32.to_nat(k), k, ds, Array.set(F64, r, k, (F64.neg(acc) / d0 : F64))))

type Weights is Type:
  Clipped{values: Array<F64>, sum: F64, valid: Bool}

def clip(n: Nat, +k: U32, next: Array<F64> & F64, q: Array<F64>, sum: F64, ok: Bool) -> Weights:
  match n:
    case 0n:
      (_, +x) = next
      +w = max(0.0, x)
      Clipped{Array.set(F64, q, k, w), (sum + w : F64), ok && finite(x)}
    case 1n+rest:
      (r, +x) = next
      +w = max(0.0, x)
      clip(rest, (k + 1 : U32), Array.get(F64, r, (58 - k : U32)),
        Array.set(F64, q, k, w), (sum + w : F64), ok && finite(x))

def normalize(n: Nat, +k: U32, +sum: F64, next: Array<F64> & F64) -> Array<F64>:
  match n:
    case 0n:
      (q, x) = next
      Array.set(F64, q, k, (x / sum : F64))
    case 1n+rest:
      (q, x) = next
      +j = (k + 1 : U32)
      normalize(rest, j, sum, Array.get(F64, Array.set(F64, q, k, (x / sum : F64)), j))

type RowResult is Type:
  RowTerm{product: Product, first_weight: F64}

def payoff_start(n: Nat, +i: U32, s: Array<F64>, next: Array<F64> & F64, start: F64) -> RowResult:
  (q, +first) = next
  RowTerm{dot(n, 0, i, 1, Array.get(F64, s, 0), (q, first), start), first}

def bounds(+n: Nat, +i: U32, +failed: F64, row: RowResult, cum: F64, lo: F64, hi: F64) -> F64 & F64:
  match n:
    case 0n:
      RowTerm{Dot{_, _, +y}, _} = row
      (min(lo, y), max(hi, y))
    case 1n+rest:
      RowTerm{Dot{s, q, +y}, first} = row
      +j = (i + 1 : U32)
      +prefix = (cum + first : F64)
      bounds(rest, j, failed, payoff_start(rest, j, s, Array.get(F64, q, j), (failed * prefix : F64)),
        prefix, min(lo, y), max(hi, y))

def support_result(result: F64 & F64) -> F64 & (F64 & U32):
  (+lo, +hi) = result
  (lo, (hi, Bool.pick(U32, gap_ok(lo, hi), 1, 2)))

def start_bounds(q: Array<F64>, s: Array<F64>, +failed: F64) -> F64 & (F64 & U32):
  support_result(bounds(59n, 0, failed, payoff_start(59n, 0, s, Array.get(F64, q, 0), (failed * 0.0 : F64)),
    0.0, 2.0, F64.neg(2.0)))

def normalize_valid(valid: Bool, sum: F64, q: Array<F64>, s: Array<F64>, failed: F64) -> F64 & (F64 & U32):
  match valid:
    case False{}:
      (F64.neg(1.0), (1.0, 2))
    case True{}:
      start_bounds(normalize(59n, 0, sum, Array.get(F64, q, 0)), s, failed)

def check_weights(checked: Weights, s: Array<F64>, failed: F64) -> F64 & (F64 & U32):
  Clipped{q, +sum, valid} = checked
  normalize_valid(valid && finite(sum) && F64.is_gt(sum, 0.0), sum, q, s, failed)

def support_start(payoffs: Payoffs, failed: F64, +d0: F64) -> F64 & (F64 & U32):
  Payoff{s, ds} = payoffs
  seed = Array.set(F64, [0.0 : F64^6n], 0, 1.0)
  r = recurrence(58n, 1, d0, recur_start(0n, 0, ds, seed))
  check_weights(clip(59n, 0, Array.get(F64, r, 59), [0.0 : F64^6n], 0.0, True{}), s, failed)

def support(valid_d0: Bool, s: Array<F64>, s0: F64, failed: F64, d0: F64) -> F64 & (F64 & U32):
  match valid_d0:
    case False{}:
      (F64.neg(1.0), (1.0, 2))
    case True{}:
      support_start(differences(58n, 0, Array.get(F64, s, 1), s0, [0.0 : F64^6n]), failed, d0)

def route(pure: Bool, lo: F64, hi: F64, +d0: F64, s: Array<F64>, s0: F64, failed: F64) -> F64 & (F64 & U32):
  match pure:
    case True{}:
      (lo, (hi, 0))
    case False{}:
      support(F64.is_ge(F64.abs(d0), (1.0 / (1000000.0 * 1000000.0) : F64)), s, s0, failed, d0)

def solve_scan(extrema: Scan, +failed: F64, +s0: F64) -> F64 & (F64 & U32):
  Extrema{s, lo, hi} = extrema
  +lower = max(lo, min(failed, s0))
  +upper = min(hi, max(failed, s0))
  route(gap_ok(lower, upper), lower, upper, (s0 - failed : F64), s, s0, failed)

def solve_first(next: Array<F64> & F64, failed: F64) -> F64 & (F64 & U32):
  (arr, +s0) = next
  solve_scan(scan(58n, 1, Array.get(F64, arr, 1), s0, s0), failed, s0)

# Rungs 0 and 1 certify a pure pair and the mirrored equalizer. Rung 2 rejects both.
def solve(values: Array<F64>, failed: F64) -> F64 & (F64 & U32):
  solve_first(Array.get(F64, values, 0), failed)
