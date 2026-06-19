#include "Graph.h"
#include "Command.h"
#include "KPlex_BB_matrix.h"
#include "2th-Reduction.h"
#include "Branch.h"
#include <new>

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
        lb = max(lb, 2 * paramK - 2);
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
