#include "bridge.h"

static _Thread_local u64 dth_alloc[3 * NCLS_ALL];

void bend_dth_init(void) {
  corpus_setup(false, 1, 0);
}

void bend_dth_thread_init(void) {
  memset(dth_alloc, 0, sizeof(dth_alloc));
}

void bend_dth_thread_close(void) {
  Env e = { CORPUS, dth_alloc };
  for (u32 cls = 0; cls < NCLS_ALL; cls++) {
    if (ALC_LEN(e, cls)) heap_hand(e, cls);
  }
}

uint32_t bend_dth_profile_id(uint32_t st, uint32_t ttd) {
  Env e = { CORPUS, dth_alloc };
  Term out[1];
  SPIN(bridge_profile_id)(e, out, st, ttd);
  return (u32)out[0];
}

uint32_t bend_dth_success(uint32_t st, uint32_t ttd, uint32_t lag) {
  Env e = { CORPUS, dth_alloc };
  Term out[1];
  SPIN(bridge_success)(e, out, st, ttd, lag);
  return (u32)out[0];
}

uint32_t bend_dth_failed(uint32_t st, uint32_t ttd) {
  Env e = { CORPUS, dth_alloc };
  Term out[1];
  SPIN(bridge_failed)(e, out, st, ttd);
  return (u32)out[0];
}

double bend_dth_revival(uint32_t st, uint32_t ttd) {
  Env e = { CORPUS, dth_alloc };
  Term out[2];
  SPIN(bridge_revival)(e, out, st, ttd);
  return dth_f64((u32)out[0], (u32)out[1]);
}

double bend_dth_failed_payoff(double x, double p) {
  Env e = { CORPUS, dth_alloc };
  Term out[2];
  SPIN(bridge_failed_payoff)(e, out, dth_f64_lo(x), dth_f64_hi(x), dth_f64_lo(p), dth_f64_hi(p));
  return dth_f64((u32)out[0], (u32)out[1]);
}

BendStage bend_dth_stage(const double success[60], double failed) {
  Env e = { CORPUS, dth_alloc };
  Term out[5];
  // An Array<F64> with 64 entries stores two U32 words per entry.
  u64 loc = heap_alloc(e, 6);
  for (u32 i = 0; i < 60; i++) {
    blk_write(e.mem, false, loc, i * 2, dth_f64_lo(success[i]));
    blk_write(e.mem, false, loc, i * 2 + 1, dth_f64_hi(success[i]));
  }
  for (u32 i = 120; i < 128; i++) blk_write(e.mem, false, loc, i, 0);
  Term input = term_blk(false, 7, loc);
  if (!SPIN(bridge_stage)(e, out, input, dth_f64_lo(failed), dth_f64_hi(failed))) {
    err_fail("DTH native stage did not return");
  }
  return (BendStage) {
    dth_f64((u32)out[0], (u32)out[1]),
    dth_f64((u32)out[2], (u32)out[3]),
    (u32)out[4],
  };
}

// Imported so the compiler retains the checked Bend entry points. The C
// sweep owns main and calls the native entry points above.
static Term dth_link_exports(Env e, Term *args, IoWork *w) {
  return term_pak(CID(Unit), 0);
}

static void __attribute__((constructor)) dth_register(void) {
  io_eff(CID(link_exports), dth_link_exports);
}
