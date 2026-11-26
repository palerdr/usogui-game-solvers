#include <stdint.h>

typedef struct {
    double lower;
    double upper;
    uint32_t rung;
} BendStage;

void bend_dth_init(void);
void bend_dth_thread_init(void);
void bend_dth_thread_close(void);
uint32_t bend_dth_profile_id(uint32_t st, uint32_t ttd);
uint32_t bend_dth_success(uint32_t st, uint32_t ttd, uint32_t lag);
uint32_t bend_dth_failed(uint32_t st, uint32_t ttd);
double bend_dth_revival(uint32_t st, uint32_t ttd);
double bend_dth_failed_payoff(double child, double revival);
BendStage bend_dth_stage(const double success[60], double failed);
