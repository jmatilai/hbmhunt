// hbmhunt: Identifies VRAM pages that trigger HBM ECC errors during read operations 
// on AMD Vega20 / MI50 (gfx906) GPUs.
//
// The program allocates nearly all available VRAM in blocks. For each data pattern, 
// it performs the following steps:
// 1. Fills every block with the current pattern.
// 2. Repeatedly reads every block and verifies the data against the pattern.
// 3. Reads each block individually to identify which blocks increment the 'umc' error counter.
// 4. Recursively divides failing blocks into 16 parts until the faulty 4 KiB pages are isolated.
//
// Once isolated, the program writes a marker into each faulty page. This allows 
// vramfind.py to determine the physical VRAM address. The program then synchronizes 
// with the external controller via files in <dir>:
// - It waits for <dir>/hunt_go to rewrite the failing pattern into the identified pages.
// - It waits for <dir>/hunt_done to perform a final GPU read and exit.
//
// Build command: 
// hipcc --offload-arch=gfx906 -O2 -std=c++17 -o hbmhunt hbmhunt.hip
//
// Usage: 
// hbmhunt [-s solid_seconds] [-r random_seconds] [-n random_seeds] <pci bdf> <dir>

#include <hip/hip_runtime.h>
#include <glob.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

#define CK(x)                                                                                    \
    do {                                                                                         \
        hipError_t e_ = (x);                                                                     \
        if (e_ != hipSuccess) {                                                                  \
            fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_));   \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

static const size_t MiB = 1 << 20, PAGE = 4096;
static const int BLOCK_READS = 10, PROBE_READS = 20, MAX_PAGES = 64;
static const int TEMP_LIMIT_C = 85;  // junction or memory

enum Kind { ZEROS, ONES, P55, PAA, RANDOM, INVERTED };
struct Pattern {
    Kind kind;
    uint32_t seed;
    std::string name;
};

static std::string sysdev, cnt_path;
static std::vector<std::string> temp_paths;  // junction, memory

static long ce() {
    long ue = -1, c = -1;
    char buf[128];
    if (FILE *f = fopen(cnt_path.c_str(), "r")) {
        while (fgets(buf, sizeof buf, f)) {
            sscanf(buf, "ue: %ld", &ue);
            sscanf(buf, "ce: %ld", &c);
        }
        fclose(f);
    }
    if (ue > 0) printf("!! uncorrectable count now %ld\n", ue);
    return c;
}

static int temp_c(int i) {
    int t = 0;
    if (FILE *f = fopen(temp_paths[i].c_str(), "r")) {
        if (fscanf(f, "%d", &t) != 1) t = 0;
        fclose(f);
    }
    return t / 1000;
}

static void check_temps() {
    int j = temp_c(0), m = temp_c(1);
    if (j > TEMP_LIMIT_C || m > TEMP_LIMIT_C) {
        printf("ABORT: junction %d C, memory %d C, limit %d C\n", j, m, TEMP_LIMIT_C);
        exit(3);
    }
}

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

__device__ inline uint64_t mix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Value of 16-byte element i (global index over all blocks).
__device__ inline u32x4 pat(int kind, uint32_t seed, uint64_t i) {
    switch (kind) {
    case ZEROS: return (u32x4)(0u);
    case ONES: return (u32x4)(~0u);
    case P55: return (u32x4)(0x55555555u);
    case PAA: return (u32x4)(0xAAAAAAAAu);
    default: {
        uint64_t x = mix(i ^ ((uint64_t)seed << 40)), y = mix(x);
        u32x4 v = u32x4{(uint32_t)x, (uint32_t)(x >> 32), (uint32_t)y, (uint32_t)(y >> 32)};
        return kind == INVERTED ? ~v : v;
    }
    }
}

__global__ void fill(u32x4 *p, size_t n, int kind, uint32_t seed, uint64_t base) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = pat(kind, seed, base + i);
}

__global__ void check(const u32x4 *p, size_t n, int kind, uint32_t seed, uint64_t base, unsigned long long *bad) {
    unsigned long long miss = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        u32x4 v = p[i], w = pat(kind, seed, base + i);
        miss += (v.x != w.x) + (v.y != w.y) + (v.z != w.z) + (v.w != w.w);
    }
    if (miss) atomicAdd(bad, miss);
}

struct Block {
    u32x4 *p;
    size_t bytes;
    uint64_t base;   // global element index of p[0]
    Pattern held;    // pattern currently stored in the block
};

static std::vector<Block> blocks;
static unsigned long long *d_bad;
static int flush_block = -1;

static int grid_for(size_t n) { return (int)std::min<size_t>((n + 255) / 256, 7680); }

static void fill_range(Block &b, size_t off, size_t len, const Pattern &pt) {
    size_t n = len / sizeof(u32x4);
    fill<<<grid_for(n), 256>>>(b.p + off / sizeof(u32x4), n, pt.kind, pt.seed, b.base + off / sizeof(u32x4));
}

static void fill_block(Block &b, const Pattern &pt) {
    fill_range(b, 0, b.bytes, pt);
    b.held = pt;
}

static void check_range(const Block &b, size_t off, size_t len) {
    size_t n = len / sizeof(u32x4);
    check<<<grid_for(n), 256>>>(b.p + off / sizeof(u32x4), n, b.held.kind, b.held.seed,
                                b.base + off / sizeof(u32x4), d_bad);
}

static unsigned long long take_bad() {
    unsigned long long h = 0;
    CK(hipMemcpy(&h, d_bad, sizeof h, hipMemcpyDeviceToHost));
    CK(hipMemset(d_bad, 0, sizeof h));
    return h;
}

// CE raised by `reads` reads of [off, off+len) in block bi. Ranges under 64 MiB are each followed by
// a read of 16 MiB of a clean block (4x the L2 size) so every read comes from HBM, not from the cache.
static long probe(int bi, size_t off, size_t len, int reads, unsigned long long *bad) {
    long before = ce();
    for (int r = 0; r < reads; r++) {
        check_range(blocks[bi], off, len);
        if (len < 64 * MiB) check_range(blocks[flush_block], 0, 16 * MiB);
    }
    CK(hipDeviceSynchronize());
    long d = ce() - before;
    *bad += take_bad();
    return d;
}

struct Page {
    int bi;
    size_t off;
    Pattern pt;
    long ce;
};

static void narrow(int bi, size_t off, size_t len, int depth, std::vector<Page> &out) {
    size_t part = std::max(len / 16, PAGE);
    for (size_t o = off; o < off + len; o += part) {
        unsigned long long bad = 0;
        long d = probe(bi, o, part, PROBE_READS, &bad);
        if (d <= 0 && !bad) continue;
        printf("%*sblock %d +0x%zx %zu KiB: CE +%ld in %d reads, mismatches %llu\n", depth * 2, "", bi, o,
               part >> 10, d, PROBE_READS, bad);
        if (part <= PAGE) {
            long again = probe(bi, o, part, PROBE_READS, &bad);
            printf("%*s  page confirmed: CE +%ld in another %d reads\n", depth * 2, "", again, PROBE_READS);
            out.push_back({bi, o, blocks[bi].held, d + again});
        } else {
            narrow(bi, o, part, depth + 1, out);
        }
        fflush(stdout);
    }
}

static void wait_file(const std::string &path) {
    struct stat st;
    printf("waiting for %s\n", path.c_str());
    fflush(stdout);
    for (int i = 0; i < 7200 && stat(path.c_str(), &st) != 0; i++) sleep(1);
}

static int select_device(const std::string &bdf) {
    int n = 0;
    CK(hipGetDeviceCount(&n));
    for (int d = 0; d < n; d++) {
        char pci[64];
        CK(hipDeviceGetPCIBusId(pci, sizeof pci, d));
        if (strcasecmp(pci, bdf.c_str()) == 0) return d;
    }
    fprintf(stderr, "no HIP device at %s\n", bdf.c_str());
    exit(1);
}

int main(int argc, char **argv) {
    int solid_s = 15, random_s = 60, seeds = 1, opt;
    while ((opt = getopt(argc, argv, "s:r:n:")) != -1) {
        if (opt == 's') solid_s = atoi(optarg);
        else if (opt == 'r') random_s = atoi(optarg);
        else if (opt == 'n') seeds = atoi(optarg);
        else return 2;
    }
    if (argc - optind != 2) {
        fprintf(stderr, "usage: %s [-s solid_s] [-r random_s] [-n seeds] <pci bdf> <dir>\n", argv[0]);
        return 2;
    }
    std::string bdf = argv[optind], dir = argv[optind + 1];
    sysdev = "/sys/bus/pci/devices/" + bdf;
    cnt_path = sysdev + "/ras/umc_err_count";
    for (const char *t : {"temp2_input", "temp3_input"}) {
        glob_t g;
        if (glob((sysdev + "/hwmon/hwmon*/" + t).c_str(), 0, nullptr, &g) != 0) {
            fprintf(stderr, "no hwmon %s for %s\n", t, bdf.c_str());
            return 1;
        }
        temp_paths.push_back(g.gl_pathv[0]);
        globfree(&g);
    }
    if (ce() < 0) { fprintf(stderr, "cannot read %s\n", cnt_path.c_str()); return 1; }

    CK(hipSetDevice(select_device(bdf)));
    size_t fr, tot;
    CK(hipMemGetInfo(&fr, &tot));
    printf("device %s, VRAM total %zu MiB, free %zu MiB, ce %ld, junction %d C, memory %d C\n", bdf.c_str(),
           tot / MiB, fr / MiB, ce(), temp_c(0), temp_c(1));

    CK(hipMalloc(&d_bad, sizeof *d_bad));
    CK(hipMemset(d_bad, 0, sizeof *d_bad));
    uint64_t base = 0;
    for (size_t sz : {256 * MiB, 32 * MiB, 4 * MiB}) {
        for (;;) {
            void *p = nullptr;
            if (hipMalloc(&p, sz) != hipSuccess) { (void)hipGetLastError(); break; }
            blocks.push_back({(u32x4 *)p, sz, base, {ZEROS, 0, ""}});
            base += sz / sizeof(u32x4);
        }
    }
    size_t total = 0;
    for (auto &b : blocks) total += b.bytes;
    CK(hipMemGetInfo(&fr, &tot));
    printf("allocated %zu blocks, %zu MiB (%.2f%% of VRAM), not allocated %zu MiB\n", blocks.size(), total / MiB,
           100.0 * total / tot, (tot - total) / MiB);

    std::vector<Pattern> patterns = {{ZEROS, 0, "zeros"}, {ONES, 0, "ones"}, {P55, 0, "0x55"}, {PAA, 0, "0xAA"}};
    for (int s = 1; s <= seeds; s++) {
        patterns.push_back({RANDOM, (uint32_t)s, "random" + std::to_string(s)});
        patterns.push_back({INVERTED, (uint32_t)s, "~random" + std::to_string(s)});
    }
    printf("%zu patterns, soak %d s solid / %d s random, about %d min\n", patterns.size(), solid_s, random_s,
           (4 * solid_s + 2 * seeds * random_s) / 60 + 1);
    fflush(stdout);

    std::map<int, std::map<std::string, long>> block_hits;  // block -> pattern -> CE
    std::vector<std::string> summary;
    for (auto &pt : patterns) {
        for (auto &b : blocks) fill_block(b, pt);
        CK(hipDeviceSynchronize());
        int secs = (pt.kind == RANDOM || pt.kind == INVERTED) ? random_s : solid_s;
        long c0 = ce();
        int passes = 0;
        double t0 = now(), tcheck = t0;
        while (now() - t0 < secs) {
            for (auto &b : blocks) check_range(b, 0, b.bytes);
            CK(hipDeviceSynchronize());
            passes++;
            if (now() - tcheck >= 1) check_temps(), tcheck = now();
        }
        // The counter is read only around the soak: each read that finds new errors logs a kernel line.
        long soak_ce = ce() - c0;
        unsigned long long bad = take_bad();
        double dt = now() - t0;
        std::vector<int> hit;
        for (size_t bi = 0; bi < blocks.size(); bi++) {
            long before = ce();
            for (int r = 0; r < BLOCK_READS; r++) check_range(blocks[bi], 0, blocks[bi].bytes);
            CK(hipDeviceSynchronize());
            long d = ce() - before;
            bad += take_bad();
            if (d > 0) {
                block_hits[(int)bi][pt.name] += d;
                hit.push_back((int)bi);
                printf("[%s]   block %zu: CE +%ld in %d reads\n", pt.name.c_str(), bi, d, BLOCK_READS);
            }
        }
        char line[512];
        snprintf(line, sizeof line,
                 "%-9s %5d passes %4.0f s %4.0f GB/s  soak CE %6ld (%.2f per pass)  hit blocks %zu  mismatches %llu"
                 "  junction %d C memory %d C",
                 pt.name.c_str(), passes, dt, passes * (double)total / dt / 1e9, soak_ce, (double)soak_ce / passes,
                 hit.size(), bad, temp_c(0), temp_c(1));
        printf("[%s] %s\n", pt.name.c_str(), line);
        summary.push_back(line);
        fflush(stdout);
    }

    printf("\nSUMMARY (one stuck bit gives exactly 1.00 CE per pass when the pattern puts the failing value in\n"
           "it, else 0.00; any other ratio means more than one bad spot or an intermittent one)\n");
    for (auto &s : summary) printf("  %s\n", s.c_str());
    if (block_hits.empty()) {
        printf("NO BAD PAGES: no block raised the error count with any pattern\n");
        return 0;
    }

    for (size_t bi = 0; bi < blocks.size() && flush_block < 0; bi++)
        if (!block_hits.count((int)bi) && blocks[bi].bytes >= 16 * MiB) flush_block = (int)bi;
    std::vector<Page> pages;
    for (auto &[bi, by_pattern] : block_hits) {
        auto best = by_pattern.begin();
        for (auto it = by_pattern.begin(); it != by_pattern.end(); ++it)
            if (it->second > best->second) best = it;
        Pattern pt = *std::find_if(patterns.begin(), patterns.end(),
                                   [&](const Pattern &p) { return p.name == best->first; });
        printf("narrowing block %d with %s (CE %ld; %zu patterns hit this block), flush block %d\n", bi,
               pt.name.c_str(), best->second, by_pattern.size(), flush_block);
        fill_block(blocks[bi], pt);
        CK(hipDeviceSynchronize());
        size_t found = pages.size();
        narrow(bi, 0, blocks[bi].bytes, 1, pages);
        if (pages.size() == found) printf("  block %d: no part raised the count on its own (intermittent?)\n", bi);
    }
    if (pages.empty()) {
        printf("NO PAGE FOUND: blocks raised the count, but narrowing found no page\n");
        return 0;
    }

    FILE *mf = fopen((dir + "/hunt_markers.txt").c_str(), "w");
    for (size_t i = 0; i < pages.size() && i < MAX_PAGES; i++) {
        char m[64] = {0};
        snprintf(m, sizeof m, "HBMHUNT:%02zu block %d off 0x%zx %s", i, pages[i].bi, pages[i].off,
                 pages[i].pt.name.c_str());
        CK(hipMemcpy((char *)blocks[pages[i].bi].p + pages[i].off, m, sizeof m, hipMemcpyHostToDevice));
        printf("marker %s\n", m);
        if (mf) fprintf(mf, "%s\n", m);
    }
    if (mf) fclose(mf);
    if (pages.size() > MAX_PAGES) printf("%zu pages found, markers only for the first %d\n", pages.size(), MAX_PAGES);
    CK(hipDeviceSynchronize());
    printf("MARKERS READY: %zu pages\n", pages.size());
    wait_file(dir + "/hunt_go");

    for (auto &p : pages) fill_range(blocks[p.bi], p.off, PAGE, p.pt);
    CK(hipDeviceSynchronize());
    printf("PAGES RESTORED\n");
    wait_file(dir + "/hunt_done");

    for (auto &p : pages) {
        unsigned long long bad = 0;
        Pattern held = blocks[p.bi].held;
        blocks[p.bi].held = p.pt;
        long d = probe(p.bi, p.off, PAGE, PROBE_READS, &bad);
        blocks[p.bi].held = held;
        printf("final GPU re-read block %d +0x%zx (%s): CE +%ld in %d reads, mismatches %llu\n", p.bi, p.off,
               p.pt.name.c_str(), d, PROBE_READS, bad);
    }
    printf("DONE\n");
    return 0;
}
