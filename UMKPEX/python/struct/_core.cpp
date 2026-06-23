#define UMKPEX_PYTHON_MODULE
#include "../../main.cpp"

#include <fcntl.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;

static py::dict result_to_dict(const UmkpexResult &result)
{
    py::dict out;
    py::dict summary;

    summary["Heur_Size"] = result.heuristic_size;
    summary["MaxKPlex_Size"] = result.max_kplex_size;
    summary["heuristic_time"] = result.heuristic_time_seconds;
    summary["ctcp_time"] = result.ctcp_time_seconds;
    summary["bnb_time"] = result.bnb_time_seconds;
    summary["t_time"] = result.total_time_seconds;

    out["summary"] = summary;
    out["Heur_Size"] = result.heuristic_size;
    out["MaxKPlex_Size"] = result.max_kplex_size;
    out["heuristic_time"] = result.heuristic_time_seconds;
    out["ctcp_time"] = result.ctcp_time_seconds;
    out["bnb_time"] = result.bnb_time_seconds;
    out["t_time"] = result.total_time_seconds;

    out["heuristic_size"] = result.heuristic_size;
    out["max_kplex_size"] = result.max_kplex_size;
    out["heuristic_time_seconds"] = result.heuristic_time_seconds;
    out["ctcp_time_seconds"] = result.ctcp_time_seconds;
    out["bnb_time_seconds"] = result.bnb_time_seconds;
    out["total_time_seconds"] = result.total_time_seconds;
    out["heuristic_is_ground_truth"] = result.heuristic_is_ground_truth;
    out["solution_vertices"] = result.solution_vertices;
    return out;
}

static py::dict solve_args_impl(const std::vector<std::string> &args, bool verbose = false)
{
    std::vector<std::string> owned_args;
    owned_args.reserve(args.size() + 1);
    owned_args.push_back("struct");
    owned_args.insert(owned_args.end(), args.begin(), args.end());

    std::vector<char *> argv;
    argv.reserve(owned_args.size());
    for (std::string &arg : owned_args)
        argv.push_back(const_cast<char *>(arg.c_str()));

    UmkpexResult result;
    int rc = 0;

    if (verbose)
    {
        rc = run_umkpex_solver(static_cast<int>(argv.size()), argv.data(), &result);
    }
    else
    {
        fflush(stdout);
        fflush(stderr);

        int stdout_fd = dup(STDOUT_FILENO);
        int stderr_fd = dup(STDERR_FILENO);
        int devnull_fd = open("/dev/null", O_WRONLY);

        if (stdout_fd == -1 || stderr_fd == -1 || devnull_fd == -1)
        {
            if (stdout_fd != -1)
                close(stdout_fd);
            if (stderr_fd != -1)
                close(stderr_fd);
            if (devnull_fd != -1)
                close(devnull_fd);
            throw std::runtime_error("Failed to redirect solver output");
        }

        dup2(devnull_fd, STDOUT_FILENO);
        dup2(devnull_fd, STDERR_FILENO);
        close(devnull_fd);

        rc = run_umkpex_solver(static_cast<int>(argv.size()), argv.data(), &result);

        fflush(stdout);
        fflush(stderr);
        dup2(stdout_fd, STDOUT_FILENO);
        dup2(stderr_fd, STDERR_FILENO);
        close(stdout_fd);
        close(stderr_fd);
    }

    if (rc != 0)
        throw std::runtime_error("UMKPEX solver failed with return code " + std::to_string(rc));
    return result_to_dict(result);
}

PYBIND11_MODULE(_core, m)
{
    m.doc() = "pybind11 bindings for the UMKPEX solver";
    m.def("solve_args", &solve_args_impl, py::arg("args"), py::arg("verbose") = false);
}
