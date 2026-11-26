#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#include <sys/sysctl.h>
#endif

#include "bridge.h"

enum {
    N = 17011, ALIVE = 16711, WIN = N, WIDTH = N + 1, LAGS = 60,
    MAX_PHI = 600, LAST = 2 * MAX_PHI
};
static const size_t table_bytes = (size_t)N * WIDTH * sizeof(double);
static double *values;
static char *pending_path;

typedef struct {
    uint32_t phi;
    uint32_t success[LAGS];
    uint32_t failed;
    double revival;
} Profile;

static Profile profiles[N];
static uint32_t ids[300][302];
static size_t bucket_start[MAX_PHI + 2];
static uint32_t by_phi[N];

typedef struct {
    size_t checker_start;
    size_t checker_end;
    uint32_t dropper;
} Job;

static Job jobs[N];
static size_t job_count;
static atomic_size_t next_job;
static atomic_bool invalid;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t ready = PTHREAD_COND_INITIALIZER;
static pthread_cond_t done = PTHREAD_COND_INITIALIZER;
static unsigned generation, running;
static bool closing;

typedef struct {
    uint64_t pure;
    uint64_t equalizer;
    double max_gap;
} Counts;

static void fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(EXIT_FAILURE);
}

static void system_failure(const char *what) {
    perror(what);
    exit(EXIT_FAILURE);
}

static void check_memory_pressure(void) {
#ifdef __APPLE__
    uint32_t pressure;
    size_t size = sizeof(pressure);
    if (sysctlbyname("kern.memorystatus_vm_pressure_level", &pressure, &size, NULL, 0) != 0)
        system_failure("check macOS memory pressure");
    if (size != sizeof(pressure) || pressure != DISPATCH_MEMORYPRESSURE_NORMAL)
        fail("Bend sweep stopped because macOS memory pressure is elevated");
#endif
}

static void remove_pending(void) {
    if (pending_path != NULL) unlink(pending_path);
}

static void interrupted(int signal) {
    const char message[] = "Bend sweep stopped before publishing the staged table\n";
    unlink(pending_path);
    write(STDERR_FILENO, message, sizeof(message) - 1);
    _exit(128 + signal);
}

static bool valid_child(uint32_t child, uint32_t phi) {
    return child == WIN || (child < N && profiles[child].phi > phi);
}

static void build_profiles(void) {
    uint32_t next = 0;
    for (uint32_t ttd = 0; ttd <= 301; ttd = ttd == 0 ? 60 : ttd + 1) {
        for (uint32_t st = 0; st < 300; ++st) {
            const uint32_t id = bend_dth_profile_id(st, ttd);
            if (id >= N) fail("Bend returned an invalid profile id");
            ids[st][ttd] = id;
            if (ttd != 301 && id >= ALIVE) continue;
            if (id != next++) fail("Bend profile order disagrees with STL");
            Profile *const p = &profiles[id];
            p->phi = st + ttd;
            p->revival = bend_dth_revival(st, ttd);
            p->failed = bend_dth_failed(st, ttd);
            for (uint32_t lag = 1; lag <= LAGS; ++lag)
                p->success[lag - 1] = bend_dth_success(st, ttd, lag);
            ++bucket_start[p->phi + 1];
        }
    }
    if (next != N) fail("Bend returned the wrong number of profiles");
    for (size_t i = 1; i <= MAX_PHI + 1; ++i) bucket_start[i] += bucket_start[i - 1];
    size_t position[MAX_PHI + 1];
    memcpy(position, bucket_start, sizeof(position));
    for (uint32_t id = 0; id < N; ++id) {
        const Profile *const p = &profiles[id];
        by_phi[position[p->phi]++] = id;
        if (!valid_child(p->failed, p->phi) || !isfinite(p->revival) ||
            p->revival < 0 || p->revival > 1)
            fail("Bend returned an invalid failed-check transition");
        for (size_t lag = 0; lag < LAGS; ++lag)
            if (!valid_child(p->success[lag], p->phi))
                fail("Bend returned an invalid successful-check transition");
    }
}

static void solve_job(const Job job, Counts *counts) {
    const double *const child_row = values + (size_t)job.dropper * WIDTH;
    for (size_t i = job.checker_start; i < job.checker_end; ++i) {
        const uint32_t checker = by_phi[i];
        const Profile *const profile = &profiles[checker];
        double success[LAGS];
        for (size_t lag = 0; lag < LAGS; ++lag) success[lag] = -child_row[profile->success[lag]];
        const double failed = bend_dth_failed_payoff(child_row[profile->failed], profile->revival);
        const BendStage stage = bend_dth_stage(success, failed);
        const double gap = stage.upper - stage.lower;
        const double value = 0.5 * (stage.lower + stage.upper);
        if (stage.rung > 1 || !isfinite(gap) || gap < 0 || gap > 1e-6 ||
            !isfinite(value) || fabs(value) > 1.0 + 1e-9) {
            if (!atomic_exchange_explicit(&invalid, true, memory_order_relaxed))
                fprintf(stderr, "uncertified Bend stage (%u,%u), rung %u, gap %.17g\n",
                        checker, job.dropper, stage.rung, gap);
            return;
        }
        values[(size_t)checker * WIDTH + job.dropper] = value;
        counts->pure += stage.rung == 0;
        counts->equalizer += stage.rung == 1;
        if (gap > counts->max_gap) counts->max_gap = gap;
    }
}

static void *worker(void *argument) {
    Counts *const counts = argument;
    unsigned seen = 0;
    bend_dth_thread_init();
    pthread_mutex_lock(&lock);
    for (;;) {
        while (!closing && generation == seen) pthread_cond_wait(&ready, &lock);
        if (closing) break;
        seen = generation;
        pthread_mutex_unlock(&lock);
        while (!atomic_load_explicit(&invalid, memory_order_relaxed)) {
            const size_t next = atomic_fetch_add_explicit(&next_job, 1, memory_order_relaxed);
            if (next >= job_count) break;
            solve_job(jobs[next], counts);
        }
        pthread_mutex_lock(&lock);
        if (--running == 0) pthread_cond_signal(&done);
    }
    pthread_mutex_unlock(&lock);
    bend_dth_thread_close();
    return NULL;
}

static double seconds(void) {
    struct timespec clock;
    if (clock_gettime(CLOCK_MONOTONIC, &clock) != 0) system_failure("clock_gettime");
    return (double)clock.tv_sec + 1e-9 * (double)clock.tv_nsec;
}

static uint64_t solve(const int stop, const unsigned threads, const double start) {
    Counts *const counts = calloc(threads, sizeof(*counts));
    pthread_t *const pool = calloc(threads, sizeof(*pool));
    if (!counts || !pool) system_failure("allocate workers");
    for (unsigned i = 0; i < threads; ++i) {
        const int status = pthread_create(&pool[i], NULL, worker, &counts[i]);
        if (status != 0) {
            errno = status;
            system_failure("pthread_create");
        }
    }
    for (int layer = LAST; layer >= stop; --layer) {
        check_memory_pressure();
        // Each job owns one Dropper column within this layer. Workers finish the
        // layer before any child values can be read by the next layer.
        job_count = 0;
        for (int a = layer > MAX_PHI ? layer - MAX_PHI : 0; a <= layer && a <= MAX_PHI; ++a) {
            const int b = layer - a;
            if (bucket_start[a] == bucket_start[a + 1]) continue;
            for (size_t j = bucket_start[b]; j < bucket_start[b + 1]; ++j)
                jobs[job_count++] = (Job){bucket_start[a], bucket_start[a + 1], by_phi[j]};
        }
        if (job_count != 0) {
            pthread_mutex_lock(&lock);
            atomic_store_explicit(&next_job, 0, memory_order_relaxed);
            running = threads;
            ++generation;
            pthread_cond_broadcast(&ready);
            while (running != 0) pthread_cond_wait(&done, &lock);
            pthread_mutex_unlock(&lock);
        }
        if (atomic_load_explicit(&invalid, memory_order_relaxed)) fail("Bend sweep failed");
        if (layer % 100 == 0 || layer == stop) {
            fprintf(stderr, "layer %4d | %.3f s\n", layer, seconds() - start);
        }
    }
    pthread_mutex_lock(&lock);
    closing = true;
    pthread_cond_broadcast(&ready);
    pthread_mutex_unlock(&lock);
    Counts total = {0};
    for (unsigned i = 0; i < threads; ++i) {
        pthread_join(pool[i], NULL);
        total.pure += counts[i].pure;
        total.equalizer += counts[i].equalizer;
        if (counts[i].max_gap > total.max_gap) total.max_gap = counts[i].max_gap;
    }
    free(pool);
    free(counts);
    fprintf(stderr, "certified %llu pure and %llu equalizer stages, max saddle gap %.17g\n",
            (unsigned long long)total.pure, (unsigned long long)total.equalizer, total.max_gap);
    return total.pure + total.equalizer;
}

static long read_integer(const char *argument, const long min, const long max) {
    char *rest;
    errno = 0;
    const long value = strtol(argument, &rest, 10);
    if (errno || rest == argument || *rest || value < min || value > max)
        fail("invalid layer or thread count");
    return value;
}

static int create_table(const char *path) {
    const size_t length = strlen(path) + sizeof(".partial.XXXXXX");
    pending_path = malloc(length);
    if (!pending_path) system_failure("malloc table path");
    snprintf(pending_path, length, "%s.partial.XXXXXX", path);
    const int file = mkstemp(pending_path);
    if (file < 0) system_failure("create staged output");
    if (atexit(remove_pending) != 0) fail("could not register output cleanup");
    struct sigaction action = {.sa_handler = interrupted};
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT, &action, NULL) != 0 ||
        sigaction(SIGTERM, &action, NULL) != 0 ||
        sigaction(SIGBUS, &action, NULL) != 0)
        system_failure("install output cleanup signals");
    const size_t size = 128 + table_bytes;
    if (ftruncate(file, (off_t)size) != 0) system_failure("size staged output");
    void *const map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, file, 0);
    if (map == MAP_FAILED) system_failure("map staged output");

    const uint16_t endian = 1;
    if (sizeof(double) != 8 || *(const unsigned char *)&endian != 1)
        fail("the NPY writer requires little-endian float64");
    unsigned char *const bytes = map;
    memcpy(bytes, "\x93NUMPY\x01\x00\x76\x00", 10);
    const char *const dict = "{'descr': '<f8', 'fortran_order': False, 'shape': (17011, 17012), }";
    const size_t dict_size = strlen(dict);
    memcpy(bytes + 10, dict, dict_size);
    memset(bytes + 10 + dict_size, ' ', 117 - dict_size);
    bytes[127] = '\n';
    values = (double *)(bytes + 128);
    return file;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) fail("usage: dth-bend OUT.npy [STOP_LAYER [THREADS]]");
    if (access(argv[1], F_OK) == 0) fail("output already exists");
    const int stop = argc >= 3 ? (int)read_integer(argv[2], 0, LAST) : 0;
    const unsigned threads = argc >= 4 ? (unsigned)read_integer(argv[3], 1, 256) : 1;
    check_memory_pressure();
    const double start = seconds();
    bend_dth_init();
    bend_dth_thread_init();
    build_profiles();
    bend_dth_thread_close();
    const int file = create_table(argv[1]);
    for (size_t i = 0; i < N; ++i) {
        if (i % 128 == 0) check_memory_pressure();
        double *const row = values + i * WIDTH;
        row[WIN] = -1.0;
        if (stop != 0)
            for (size_t j = 0; j < N; ++j)
                if (profiles[i].phi + profiles[j].phi < (unsigned)stop) row[j] = NAN;
    }
    const uint64_t visited = solve(stop, threads, start);
    if (stop == 0) {
        if (visited != (uint64_t)N * N) fail("incomplete full sweep");
        fprintf(stderr, "root %.17g\n", values[0]);
    }
    if (stop <= 1082) {
        const size_t anchor = ids[240][0];
        fprintf(stderr, "dead-band anchor %.17g\n", values[anchor * WIDTH + anchor]);
    }
    if (msync((unsigned char *)values - 128, 128 + table_bytes, MS_SYNC) != 0 ||
        fsync(file) != 0 || close(file) != 0) system_failure("flush output");
    if (link(pending_path, argv[1]) != 0) system_failure("publish output");
    fprintf(stderr, "wrote %s (%.3f s)\n", argv[1], seconds() - start);
}
