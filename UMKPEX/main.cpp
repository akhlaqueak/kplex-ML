#include "Graph.h"
#include "Command.h"
#include "KPlex_BB_matrix.h"
#include "2th-Reduction.h"
#include "Branch.h"
#include "MLPredictor.h"
#include "MLAlgoConfig.h"
#include <new>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>
#ifdef __linux__
#include <sched.h>
#endif

set<ui> solution;

string file_path;
Graph g;
double algorithm_start_time = 0, total_heuris_time = 0;
double FastHeuris_time = 0;
double StrongHeuris_time = 0;
double strong_reduce_time = 0;
int FastHeuris_lb;
double heuristic_time, ctcp_time, bnb_time;

struct UmkpexResult
{
    int heuristic_size = 0;
    int max_kplex_size = 0;
    double heuristic_time_seconds = 0.0;
    double ctcp_time_seconds = 0.0;
    double bnb_time_seconds = 0.0;
    double total_time_seconds = 0.0;
    bool heuristic_is_ground_truth = false;
    vector<ui> solution_vertices;
};

static string read_cpu_model_name()
{
#ifdef __linux__
    ifstream cpuinfo("/proc/cpuinfo");
    string line;
    while (getline(cpuinfo, line))
    {
        const string key = "model name";
        if (line.compare(0, key.size(), key) == 0)
        {
            size_t pos = line.find(':');
            if (pos != string::npos)
            {
                string model = line.substr(pos + 1);
                while (!model.empty() && model.front() == ' ')
                    model.erase(model.begin());
                return model;
            }
        }
    }
#endif
    return "unknown";
}

static string get_arch_class(const string &cpu_model)
{
    if (cpu_model.find("6248R") != string::npos)
        return "intel-6248r";
    if (cpu_model.find("7713") != string::npos)
        return "amd-7713";
    if (cpu_model.find("Intel") != string::npos)
        return "intel-other";
    if (cpu_model.find("AMD") != string::npos)
        return "amd-other";
    return "unknown";
}

static string get_affinity_summary()
{
#ifdef __linux__
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0)
    {
        int count = 0;
        vector<int> cpus;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
        {
            if (CPU_ISSET(cpu, &mask))
            {
                ++count;
                cpus.push_back(cpu);
            }
        }
        if (!cpus.empty())
        {
            return "count=" + to_string(count) +
                   " first=" + to_string(cpus.front()) +
                   " last=" + to_string(cpus.back());
        }
        return "count=0";
    }
#endif
    return "unknown";
}

static long long read_meminfo_kb(const string &key)
{
#ifdef __linux__
    ifstream meminfo("/proc/meminfo");
    string label;
    long long value = -1;
    string unit;
    while (meminfo >> label >> value >> unit)
    {
        if (label == key + ":")
            return value;
    }
#endif
    return -1;
}

static long long read_self_rss_kb()
{
#ifdef __linux__
    ifstream status("/proc/self/status");
    string label;
    while (status >> label)
    {
        if (label == "VmRSS:")
        {
            long long value = -1;
            string unit;
            status >> value >> unit;
            return value;
        }
        string rest;
        getline(status, rest);
    }
#endif
    return -1;
}

static double read_loadavg_1m()
{
#ifdef __linux__
    ifstream loadavg("/proc/loadavg");
    double load1 = -1.0;
    if (loadavg >> load1)
        return load1;
#endif
    return -1.0;
}

static double read_current_cpu_mhz()
{
#ifdef __linux__
    ifstream cpuinfo("/proc/cpuinfo");
    string line;
    while (getline(cpuinfo, line))
    {
        const string key = "cpu MHz";
        if (line.compare(0, key.size(), key) == 0)
        {
            size_t pos = line.find(':');
            if (pos != string::npos)
            {
                string value = line.substr(pos + 1);
                while (!value.empty() && value.front() == ' ')
                    value.erase(value.begin());
                return atof(value.c_str());
            }
        }
    }
#endif
    return -1.0;
}

static long long read_self_status_kb_or_count(const string &key)
{
#ifdef __linux__
    ifstream status("/proc/self/status");
    string label;
    while (status >> label)
    {
        if (label == key + ":")
        {
            long long value = -1;
            status >> value;
            return value;
        }
        string rest;
        getline(status, rest);
    }
#endif
    return -1;
}

static string read_numa_mem_available_summary()
{
#ifdef __linux__
    vector<string> parts;
    for (int node = 0;; ++node)
    {
        string path = "/sys/devices/system/node/node" + to_string(node) + "/meminfo";
        ifstream meminfo(path);
        if (!meminfo.is_open())
            break;

        string line;
        long long memfree_kb = -1;
        while (getline(meminfo, line))
        {
            if (line.find("MemFree") != string::npos)
            {
                stringstream ss(line);
                string token;
                long long value = -1;
                while (ss >> token)
                {
                    bool all_digits = !token.empty();
                    for (char ch : token)
                        all_digits = all_digits && isdigit(static_cast<unsigned char>(ch));
                    if (all_digits)
                        value = atoll(token.c_str());
                }
                memfree_kb = value;
                break;
            }
        }
        if (memfree_kb >= 0)
        {
            parts.push_back("n" + to_string(node) + "=" +
                            to_string(memfree_kb / 1024) + "MB");
        }
    }
    if (!parts.empty())
    {
        string summary;
        for (size_t i = 0; i < parts.size(); ++i)
        {
            if (i)
                summary += ",";
            summary += parts[i];
        }
        return summary;
    }
#endif
    return "unknown";
}

static void print_runtime_diag()
{
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) != 0)
        strcpy(hostname, "unknown");
    hostname[sizeof(hostname) - 1] = '\0';

    string cpu_model = read_cpu_model_name();
    string arch_class = get_arch_class(cpu_model);
    long long mem_available_kb = read_meminfo_kb("MemAvailable");
    long long mem_total_kb = read_meminfo_kb("MemTotal");
    long long rss_kb = read_self_rss_kb();
    double load1 = read_loadavg_1m();
    double cpu_mhz = read_current_cpu_mhz();
    long long threads = read_self_status_kb_or_count("Threads");
    long long ctxt_vol = read_self_status_kb_or_count("voluntary_ctxt_switches");
    long long ctxt_nonvol = read_self_status_kb_or_count("nonvoluntary_ctxt_switches");
    string numa_mem = read_numa_mem_available_summary();

    printf("Runtime diag: host=%s pid=%d arch=%s cpu_model=%s affinity=%s rss_mb=%.1f mem_avail_gb=%.1f mem_total_gb=%.1f load1=%.2f cpu_mhz=%.1f threads=%lld ctxt_v=%lld ctxt_nv=%lld numa_free=%s\n",
           hostname,
           (int)getpid(),
           arch_class.c_str(),
           cpu_model.c_str(),
           get_affinity_summary().c_str(),
           rss_kb >= 0 ? rss_kb / 1024.0 : -1.0,
           mem_available_kb >= 0 ? mem_available_kb / 1024.0 / 1024.0 : -1.0,
           mem_total_kb >= 0 ? mem_total_kb / 1024.0 / 1024.0 : -1.0,
           load1,
           cpu_mhz,
           threads,
           ctxt_vol,
           ctxt_nonvol,
           numa_mem.c_str());
    fflush(stdout);
}

static void reset_solver_state()
{
    solution.clear();
    file_path.clear();
    g.~Graph();
    new (&g) Graph();
    algorithm_start_time = 0;
    total_heuris_time = 0;
    FastHeuris_time = 0;
    StrongHeuris_time = 0;
    strong_reduce_time = 0;
    FastHeuris_lb = 0;
    heuristic_time = 0;
    ctcp_time = 0;
    bnb_time = 0;
    lb = 0;
    paramK = 0;
    list_triangle_time = 0;
    after_maxkp_time = 0;
    reduce_kPlexT_time_matrix = 0;
}

static void fill_result(UmkpexResult &result, ui heur_size, double local_bnb_time_seconds)
{
    result.heuristic_size = static_cast<int>(heur_size);
    result.max_kplex_size = static_cast<int>(solution.size());
    result.heuristic_time_seconds = StrongHeuris_time / 1e6;
    result.ctcp_time_seconds = FastHeuris_time / 1e6;
    result.bnb_time_seconds = local_bnb_time_seconds;
    result.total_time_seconds = (get_system_time_microsecond() - algorithm_start_time) / 1e6;
    result.heuristic_is_ground_truth = (g.n == 0 || g.n <= static_cast<ui>(lb));
    result.solution_vertices.assign(solution.begin(), solution.end());
}

void print_solution()
{
    if (solution.size() < 2 * paramK - 1)
    {
        printf("***We can't find a plex larger than 2k-2!! The following is a heuristic solution.\n");
        if (solution.size())
        {
            print_set(solution);
        }
        return;
    }
#ifndef NO_DUMP
    printf("Maximum solution(size= %d ):\n", (int)solution.size());
    print_set(solution);
#endif
    fflush(stdout);
}

bool print_heuris_log(UmkpexResult *result = nullptr)
{
    puts("*************Heuristic result*************");
    after_maxkp_time = get_system_time_microsecond();

    total_heuris_time = get_system_time_microsecond() - algorithm_start_time;
    printf("list triangles time: %.4lf s, strong reduce time: %.4lf s\n", list_triangle_time / 1e6, strong_reduce_time / 1e6);
    printf("total-heuristic-time= %.4lf s, FastHeuris-time= %.4lf s, StrongHeuris-time= %.4lf s\n",
           total_heuris_time / 1e6, FastHeuris_time / 1e6, StrongHeuris_time / 1e6);
    printf("lb= %d , FastHeuris-lb= %d \n", lb, FastHeuris_lb);
    if (solution.size() >= 2 * paramK - 1)
        assert(solution.size() == lb);

    if (g.n <= lb)
    {
        printf("The heuristic solution is the ground truth!\n");
        print_solution();
        puts("------------------{whole procedure: kPEX}---------------------");
        printf("\n>>Heur_Size: %u MaxKPlex_Size: %u heuristic_time: %.4lf ctcp_time: %.4lf bnb_time: prep t_time: %.4lf\n",
               lb, (ui)solution.size(), StrongHeuris_time / 1e6, FastHeuris_time / 1e6,
               (get_system_time_microsecond() - algorithm_start_time) / 1e6);
        if (result != nullptr)
            fill_result(*result, lb, 0.0);
        return true;
    }
    return false;
}

bool strong_reduce(UmkpexResult *result = nullptr)
{
    Timer start_strong_reduce;
    Reduction reduce(&g);
    reduce.strong_reduce(lb);
    printf("Afer strong reduce, n= %u , m= %u, use time %.4lf s\n", g.n, g.m / 2, start_strong_reduce.get_time() / 1e6);
    strong_reduce_time += start_strong_reduce.get_time();

    if (lb >= g.n)
    {
        g.n = 0;
        return print_heuris_log(result);
    }
    return false;
}

bool FastHeuris(UmkpexResult *result = nullptr)
{
    Timer t("FastHeuris");
    {
#ifndef NO_SQRT
        lb = g.sqrt_degeneracy(&solution);
        printf("sqrt lb= %d, use time %.4lf s\n", lb, t.get_time() / 1e6);
#endif
        lb = max(lb, g.degeneracy_and_reduce(lb, &solution));
        printf("After degeneracy and weak reduce, n= %u , m= %u , lb= %d , use time %.4lf s\n", g.n, g.m / 2, lb, t.get_time() / 1e6);
        if (lb >= g.n)
        {
            g.n = 0;
            FastHeuris_lb = lb;
            FastHeuris_time = t.get_time();
            return print_heuris_log(result);
        }
    }
    FastHeuris_lb = lb;
    FastHeuris_time = t.get_time();
    return false;
}

bool StrongHeuris(UmkpexResult *result = nullptr)
{
    int iteration_cnt = 1;
    double time_limit = 0;
    if (PARA.heuristic == "Both-Heur")
    {
        time_limit = max(FastHeuris_time, 10e6);
    }
    Timer t_extend("StrongHeuris");
    while (1)
    {
#ifdef NO_STRONG_HEURIS
        break;
#endif
        int extend_lb = 0;
        if (solution.size() < 2 * paramK - 2)
        {
            break;
        }
        extend_lb = g.strong_heuris(lb, solution, time_limit);
        printf("%dth-StrongHeuris lb= %d\n", iteration_cnt++, extend_lb);
        if (extend_lb <= lb)
            break;
        lb = extend_lb;
        g.weak_reduce(lb);

        Timer start_strong_reduce;
        Reduction reduce(&g);
        reduce.strong_reduce(lb);
        printf("Afer strong reduce, n= %u , m= %u, use time %.4lf s\n", g.n, g.m / 2, start_strong_reduce.get_time() / 1e6);

        if (lb >= g.n)
        {
            g.n = 0;
            StrongHeuris_time += t_extend.get_time();
            return print_heuris_log(result);
        }
    }
    StrongHeuris_time += t_extend.get_time();
    return false;
}

bool EgoHeuris(UmkpexResult *result = nullptr)
{
    Timer t("ego-degen");
    int ego_lb = g.ego_degen(&solution);

    StrongHeuris_time = t.get_time();
    cout << "ego degen lb: " << ego_lb << " use time: " << t.get_time_seconds() << endl;
    if (ego_lb > lb)
    {
        lb = ego_lb;
        g.weak_reduce(lb);
        Timer weak_reduce_timer;
        printf("Afer weak reduce, n= %u , m= %u, use time %.4lf s\n", g.n, g.m / 2, weak_reduce_timer.get_time() / 1e6);
    }
    return strong_reduce(result);
}

bool heuris(UmkpexResult *result = nullptr)
{
    if (FastHeuris(result))
        return true;
    if (PARA.heuristic == "EGO-Heur" || PARA.heuristic == "Both-Heur")
    {
        if (EgoHeuris(result))
            return true;
        if (PARA.heuristic == "EGO-Heur")
            return false;
    }
    else if (strong_reduce(result))
    {
        return true;
    }
    return StrongHeuris(result);
}

void bnb()
{
    Graph_reduced *G;
    G = new Graph_reduced_adjacent_list(g);
    Branch branch(*G, lb);
    if (PARA.framework == "oriented")
        branch.IE_framework_oriented();
    else
        branch.IE_framework();
    if (solution.size() < branch.solution.size())
    {
        solution.clear();
        for (int v : branch.solution)
            solution.insert(v);
    }
    delete G;
}

struct RaceOption
{
    string label;
    CommandLine para; // fully-resolved config this option's thread should run with
};

/**
 * @brief Apply one "key=value" override onto a race option's config, validating
 * against the same value lists CommandLine's constructor uses. framework, algorithm,
 * branching, bounding and ctcp may all vary independently per race option because
 * they're only consulted during the B&B phase (see Branch.h / KPlex_BB_matrix.h /
 * 2th-Reduction.h); heuristic cannot, because by the time racing starts the shared
 * heuristic phase has already run once, before any option-specific config exists.
 */
static bool apply_race_override(CommandLine &cfg, const string &key, const string &value, string &err)
{
    auto invalid = [&](const vector<string> &valid)
    {
        err = "invalid value '" + value + "' for '" + key + "'; valid: ";
        for (auto &v : valid)
            err += v + " ";
        return false;
    };
    if (key == "framework")
    {
        auto valid = CommandLine::valid_frameworks();
        if (find(valid.begin(), valid.end(), value) == valid.end())
            return invalid(valid);
        cfg.framework = value;
    }
    else if (key == "algorithm")
    {
        auto valid = CommandLine::valid_algorithms();
        if (find(valid.begin(), valid.end(), value) == valid.end())
            return invalid(valid);
        cfg.algorithm = value;
    }
    else if (key == "branching")
    {
        auto valid = CommandLine::valid_branchings();
        if (find(valid.begin(), valid.end(), value) == valid.end())
            return invalid(valid);
        cfg.branching = value;
    }
    else if (key == "bounding")
    {
        auto valid = CommandLine::valid_boundings();
        if (find(valid.begin(), valid.end(), value) == valid.end())
            return invalid(valid);
        cfg.bounding = value;
    }
    else if (key == "ctcp")
    {
        auto valid = CommandLine::valid_ctcp_methods();
        if (find(valid.begin(), valid.end(), value) == valid.end())
            return invalid(valid);
        cfg.ctcp = value;
    }
    else if (key == "UBR2")
    {
        cfg.UBR2 = (value == "true");
    }
    else if (key == "heuristic")
    {
        err = "'heuristic' can't be raced: the heuristic phase already ran once, "
              "in common, before any race option exists";
        return false;
    }
    else
    {
        err = "unknown race key '" + key + "' (valid: framework, algorithm, branching, bounding, ctcp, UBR2)";
        return false;
    }
    return true;
}

/**
 * @brief Parse a -race spec into fully-resolved per-option configs.
 *
 * Format: options separated by ';', each an optional "label:" followed by
 * comma-separated "key=value" overrides layered onto the already-parsed, shared
 * `base` config, e.g.:
 *   -race "fast:framework=oriented,branching=Pivot-Br;exhaustive:framework=default,branching=S-Br"
 * A bare value list without '=' (as previously supported by -race-framework) still
 * works as shorthand for "framework=<value>", e.g. "-race default;oriented".
 */
static vector<RaceOption> parse_race_options(const string &spec, const CommandLine &base)
{
    vector<RaceOption> options;
    if (spec.empty())
        return options;

    vector<string> tokens;
    {
        stringstream ss(spec);
        string tok;
        while (getline(ss, tok, ';'))
            if (!tok.empty())
                tokens.push_back(tok);
    }

    int auto_idx = 0;
    for (auto &tok : tokens)
    {
        string label, body;
        size_t colon = tok.find(':');
        size_t eq = tok.find('=');
        if (colon != string::npos && (eq == string::npos || colon < eq))
        {
            label = tok.substr(0, colon);
            body = tok.substr(colon + 1);
        }
        else
        {
            body = tok;
        }
        if (label.empty())
            label = "option" + to_string(auto_idx);
        auto_idx++;

        CommandLine cfg = base;
        bool ok = true;
        if (body.find('=') == string::npos)
        {
            // shorthand: a bare value means framework=<value>
            string err;
            if (!apply_race_override(cfg, "framework", body, err))
            {
                printf("Race option '%s': %s\n", label.c_str(), err.c_str());
                ok = false;
            }
        }
        else
        {
            stringstream bs(body);
            string kv;
            while (getline(bs, kv, ','))
            {
                if (kv.empty())
                    continue;
                size_t kv_eq = kv.find('=');
                if (kv_eq == string::npos)
                {
                    printf("Race option '%s': ignoring malformed override '%s' (expected key=value)\n",
                           label.c_str(), kv.c_str());
                    continue;
                }
                string key = kv.substr(0, kv_eq), value = kv.substr(kv_eq + 1);
                string err;
                if (!apply_race_override(cfg, key, value, err))
                {
                    printf("Race option '%s': %s\n", label.c_str(), err.c_str());
                    ok = false;
                }
            }
        }
        if (ok)
            options.push_back({label, cfg});
    }

    if (options.size() < 2)
    {
        if (!options.empty())
            printf("-race needs at least 2 valid options to race; ignoring\n");
        options.clear();
    }
    return options;
}

/**
 * @brief Race several fully-independent configs (any mix of framework, algorithm,
 * branching, bounding, ctcp) in parallel threads on independent copies of the
 * already-preprocessed graph `g`. Each thread sets its own thread_local PARA before
 * doing anything else, so one thread's "Adaptive-Br" resolution or bounding/branching
 * choice can never be observed by another. Since all options solve the identical
 * exact problem to completion, whichever thread finishes first has the proven-optimal
 * answer: the others are signalled to abort (via Branch::stop_flag, checked at every
 * search-tree node) and joined before the parent reports the winner's result.
 *
 * `options` may hold more entries than are meant to run at once: if `batch_size` is
 * positive and smaller than options.size(), only the first `batch_size` options are
 * started immediately. If none of the running options has won after
 * `escalate_timeout_sec` seconds, the next `batch_size` options are started alongside
 * the still-running ones (they all share the same stop_flag/winner, so late starters
 * race the earlier ones exactly as if they'd all started together); this repeats until
 * a winner is found or `options` is exhausted, at which point the wait becomes
 * unbounded. batch_size == 0 (or >= options.size()) starts every option immediately,
 * matching the original non-escalating behavior used by manual -race specs.
 */
void run_parallel_bnb(const vector<RaceOption> &options, size_t batch_size = 0, int escalate_timeout_sec = 0)
{
    int n_opt = (int)options.size();
    vector<set<ui>> outcomes(n_opt);
    atomic<bool> stop_flag(false);
    atomic<int> winner(-1);
    mutex done_mutex;
    condition_variable done_cv;
    set<ui> heur_solution_snapshot = solution; // heuristic's answer, same for every option

    vector<thread> workers;
    auto launch = [&](int i)
    {
        workers.emplace_back([&, i]()
        {
            PARA = options[i].para; // this thread's own thread_local PARA, independent of all others

            Graph_reduced *G = new Graph_reduced_adjacent_list(g);
            Branch branch(*G, lb);
            branch.verbose = false;
            branch.stop_flag = &stop_flag;

            if (PARA.framework == "oriented")
                branch.IE_framework_oriented();
            else
                branch.IE_framework();
            delete G;

            if (stop_flag.load(memory_order_relaxed))
                return; // another option already won; this search was aborted mid-way

            set<ui> final_solution = heur_solution_snapshot;
            if (branch.solution.size() > final_solution.size())
            {
                final_solution.clear();
                for (int v : branch.solution)
                    final_solution.insert(v);
            }
            outcomes[i] = final_solution;

            int expected = -1;
            if (winner.compare_exchange_strong(expected, i))
            {
                stop_flag.store(true, memory_order_relaxed);
                lock_guard<mutex> lk(done_mutex);
                done_cv.notify_all();
            }
        });
    };

    auto announce_and_launch = [&](int from, int to)
    {
        printf("Race: starting %d option(s):", to - from);
        for (int i = from; i < to; i++)
            printf(" '%s'", options[i].label.c_str());
        printf("\n");
        for (int i = from; i < to; i++)
            launch(i);
    };

    size_t round_size = (batch_size == 0 || batch_size >= (size_t)n_opt) ? (size_t)n_opt : batch_size;
    int launched = 0;
    announce_and_launch(launched, (int)round_size);
    launched = (int)round_size;

    {
        unique_lock<mutex> lk(done_mutex);
        while (winner.load(memory_order_relaxed) == -1 && launched < n_opt)
        {
            if (done_cv.wait_for(lk, chrono::seconds(escalate_timeout_sec),
                                  [&] { return winner.load(memory_order_relaxed) != -1; }))
                break; // a winner showed up during this round

            int round_end = min(n_opt, launched + (int)round_size);
            printf("ML race: no winner after %ds with %d option(s) running -- adding %d more\n",
                   escalate_timeout_sec, launched, round_end - launched);
            announce_and_launch(launched, round_end);
            launched = round_end;
        }
        done_cv.wait(lk, [&] { return winner.load(memory_order_relaxed) != -1; });
    }
    for (auto &t : workers)
        t.join();

    int w = winner.load();
    const CommandLine &won = options[w].para;
    printf("Race winner: '%s' (framework=%s algorithm=%s branching=%s bounding=%s ctcp=%s) out of %d option(s) started (of %d total)\n",
           options[w].label.c_str(), won.framework.c_str(), won.algorithm.c_str(),
           won.branching.c_str(), won.bounding.c_str(), won.ctcp.c_str(), launched, n_opt);
    solution = outcomes[w];
}

#ifdef USE_ML_PREDICTOR

/**
 * @brief Per-label CommandLine overrides, reusing the same "key=value" override
 * mechanism -race uses (apply_race_override above). The actual mapping table lives
 * in MLAlgoConfig.h -- edit that file, not this one, to fill in per-algorithm configs.
 */
static CommandLine ml_label_to_commandline(const string &label, const CommandLine &base)
{
    CommandLine cfg = base;
    const auto &overrides_table = load_ml_algo_config(base.ml_algo_config_path);
    auto it = overrides_table.find(label);
    if (it == overrides_table.end())
    {
        printf("ML predictor: unrecognized algorithm label '%s', falling back to base config\n", label.c_str());
        return cfg;
    }
    for (auto &kv : it->second)
    {
        string err;
        if (!apply_race_override(cfg, kv.first, kv.second, err))
            printf("ML predictor: label '%s': %s\n", label.c_str(), err.c_str());
    }
    return cfg;
}

/**
 * @brief Build race options from the saved model's full ranking of algorithms for the
 * current, already-preprocessed graph `g` -- the ML-driven alternative to a manual
 * -race spec. Returns every rankable algorithm (not just one batch): run_parallel_bnb
 * is the one that decides, via base.ml_batch_size/ml_escalate_timeout_sec, how many of
 * these to start at once and when to add more.
 */
static vector<RaceOption> build_ml_race_options(const Graph &graph, int paramK_, const CommandLine &base)
{
    vector<RaceOption> options;
    try
    {
        GraphMLFeatures feats = extract_ml_features(graph, paramK_);
        printf("ML predictor: features n=%.0f m=%.0f d_avg=%.3f d_max=%.0f core_max=%.0f core_min=%.0f "
               "shell_num=%.0f shell_avgSize=%.2f shell_maxSize=%.0f shell_minSize=%.0f shell_stdSize=%.2f\n",
               feats.n, feats.m, feats.d_avg, feats.d_max, feats.core_max, feats.core_min,
               feats.shell_num, feats.shell_avgSize, feats.shell_maxSize, feats.shell_minSize, feats.shell_stdSize);

        Scheme1Predictor &predictor = Scheme1Predictor::instance(base.ml_model_path, base.ml_meta_path);
        auto ranked = predictor.predict_ranked(feats);

        for (auto &pr : ranked)
        {
            printf("ML predictor: candidate '%s' prob=%.4f\n", pr.first.c_str(), pr.second);
            options.push_back({pr.first, ml_label_to_commandline(pr.first, base)});
        }
    }
    catch (const std::exception &e)
    {
        printf("ML predictor: %s -- falling back to single bnb()\n", e.what());
        return {};
    }

    if (options.size() < 2)
    {
        printf("ML predictor: fewer than 2 usable race options, falling back to single bnb()\n");
        options.clear();
    }
    return options;
}

#endif // USE_ML_PREDICTOR

int run_umkpex_solver(int argc, char *argv[], UmkpexResult *result = nullptr)
{
    reset_solver_state();
    PARA = CommandLine(argc, argv);
    if (argc < 3)
    {
        printf("2 params are required !!! \n");
        printf("usage: ./kPEX graph_path k\n");
        return 1;
    }
    file_path = string(argv[1]);
    paramK = atoi(argv[2]);
    g.readFromFile(file_path);
    print_runtime_diag();

    algorithm_start_time = get_system_time_microsecond();

    puts("------------------{start KPHeuris}---------------------");
    Timer prepro("heuristic and preprocess");
    if (heuris(result))
        return 0;
    prepro.print_time();
    print_heuris_log();

    ui heur_size = solution.size();
    puts("------------------{start BRB_Rec}---------------------");
    Timer bnb_timer;
    vector<RaceOption> race_options = parse_race_options(PARA.race, PARA);
    bool ml_race_active = false;
#ifdef USE_ML_PREDICTOR
    if (race_options.empty() && PARA.ml_race)
    {
        race_options = build_ml_race_options(g, paramK, PARA);
        ml_race_active = !race_options.empty();
    }
#else
    if (PARA.ml_race)
        printf("Warning: -ml-race requested but this binary was built without ML support "
               "(build with `make ml`; see makefile and MLPredictor.h). Falling back to single bnb().\n");
#endif
    if (!race_options.empty())
    {
        if (ml_race_active)
            run_parallel_bnb(race_options, (size_t)PARA.ml_batch_size, PARA.ml_escalate_timeout_sec);
        else
            run_parallel_bnb(race_options);
    }
    else
        bnb();

    print_solution();

    puts("------------------{whole procedure: kPEX}---------------------");
    printf("\n>>Heur_Size: %u MaxKPlex_Size: %u heuristic_time: %.4lf ctcp_time: %.4lf bnb_time: %.4lf t_time: %.4lf\n",
           heur_size, (ui)solution.size(), StrongHeuris_time / 1e6, FastHeuris_time / 1e6,
           bnb_timer.get_time_seconds(), (get_system_time_microsecond() - algorithm_start_time) / 1e6);

    if (result != nullptr)
        fill_result(*result, heur_size, bnb_timer.get_time_seconds());
    return 0;
}

#ifndef UMKPEX_PYTHON_MODULE
int main(int argc, char *argv[])
{
    return run_umkpex_solver(argc, argv, nullptr);
}
#endif
