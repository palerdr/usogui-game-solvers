import Base
import ./profiles.bend as Profiles
import ./stage.bend as Stage

def bridge_profile_id(st: U32, ttd: U32) -> U32:
  Profiles.profile_id(st, ttd)

def bridge_success(st: U32, ttd: U32, lag: U32) -> U32:
  Profiles.success(st, ttd, lag)

def bridge_failed(st: U32, ttd: U32) -> U32:
  Profiles.failed(st, ttd)

def bridge_revival(st: U32, ttd: U32) -> F64:
  Profiles.revival(st, ttd)

def bridge_failed_payoff(child: F64, p: F64) -> F64:
  Profiles.failed_payoff(child, p)

def bridge_stage(s: Array<F64>, failed: F64) -> F64 & (F64 & U32):
  Stage.solve(s, failed)

def link_exports() -> IO(Unit):
  import "./bridge.c"

def main() -> IO(Unit):
  link_exports()
