#ifndef MLPREDICTOR_H
#define MLPREDICTOR_H

/**
 * ML-driven algorithm selection for the parallel race (see run_parallel_bnb in
 * main.cpp). Everything in this file is compiled only when built with
 * -DUSE_ML_PREDICTOR (see makefile's `ml` target), since it links libxgboost.
 *
 * Pipeline: extract_ml_features() reads the already-preprocessed graph `g` (as it
 * stands right before the B&B phase) and computes the same 12 features the Scheme 1
 * XGBoost model was trained on (restructured_pipeline/src/features.py::feature_cols).
 * Scheme1Predictor then loads a booster exported by
 * restructured_pipeline/src/export_xgb_booster.py (plain XGBoost model.json, no
 * sklearn/Python dependency at inference time) and ranks the 24 algorithm-label
 * classes by predicted probability.
 *
 * This module is read-only with respect to `g`: the k-core decomposition below is a
 * self-contained O(n+m) Batagelj-Zaversnik peel, independent of the solver's own
 * degeneracy routines in Graph.h, so it never shares mutable state with the solver.
 */
#ifdef USE_ML_PREDICTOR

#include "Graph.h"
#include <xgboost/c_api.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define XGB_CHECK(call)                                                            \
    do                                                                             \
    {                                                                             \
        int _xgb_status = (call);                                                 \
        if (_xgb_status != 0)                                                     \
            throw std::runtime_error(std::string("XGBoost error: ") + XGBGetLastError()); \
    } while (0)

struct GraphMLFeatures
{
    double k_value = 0, n = 0, m = 0, d_avg = 0, d_max = 0, core_max = 0, core_min = 0;
    double shell_num = 0, shell_avgSize = 0, shell_maxSize = 0, shell_minSize = 0, shell_stdSize = 0;
};

/**
 * @brief Standard O(n+m) k-core decomposition (Batagelj & Zaversnik, 2003; the same
 * algorithm networkx's core_number uses). Returns core[v] = core number of vertex v
 * on the CSR adjacency (g.pstart/g.edge_to/g.d) as it currently stands.
 */
inline vector<ui> compute_core_numbers(const Graph &g)
{
    ui n = g.n;
    vector<ui> deg(n);
    ui max_deg = 0;
    for (ui v = 0; v < n; v++)
    {
        deg[v] = g.d[v];
        max_deg = max(max_deg, deg[v]);
    }

    // bin_start[d] = number of vertices with degree < d, i.e. the start offset of
    // degree-d's bucket in the degree-sorted `vert` array built below.
    vector<ui> bin_start(max_deg + 2, 0);
    for (ui v = 0; v < n; v++)
        bin_start[deg[v] + 1]++;
    for (ui d = 1; d <= max_deg + 1; d++)
        bin_start[d] += bin_start[d - 1];

    vector<ui> pos(n), vert(n);
    vector<ui> next_free = bin_start; // placement cursor; bin_start itself stays untouched
    for (ui v = 0; v < n; v++)
    {
        pos[v] = next_free[deg[v]];
        vert[pos[v]] = v;
        next_free[deg[v]]++;
    }

    vector<ui> core(n, 0);
    vector<ui> deg_mut = deg;
    for (ui i = 0; i < n; i++)
    {
        ui v = vert[i];
        core[v] = deg_mut[v];
        for (ui j = g.pstart[v]; j < g.pstart[v + 1]; j++)
        {
            ui u = g.edge_to[j];
            if (deg_mut[u] > deg_mut[v])
            {
                ui du = deg_mut[u];
                ui pu = pos[u];
                ui pw = bin_start[du]; // current front of u's bucket
                ui w = vert[pw];
                if (u != w)
                {
                    vert[pu] = w;
                    pos[w] = pu;
                    vert[pw] = u;
                    pos[u] = pw;
                }
                bin_start[du]++;
                deg_mut[u]--;
            }
        }
    }
    return core;
}

/**
 * @brief Compute the 12 Scheme 1 features (feature_cols order in features.py) from
 * the graph `g` as it stands right now (post-heuristic/preprocessing, pre-B&B) and
 * the current k parameter.
 */
inline GraphMLFeatures extract_ml_features(const Graph &g, int paramK)
{
    GraphMLFeatures f;
    f.k_value = static_cast<double>(paramK);
    f.n = static_cast<double>(g.n);
    f.m = static_cast<double>(g.m) / 2.0; // g.m double-counts each undirected edge
    f.d_avg = g.n > 0 ? static_cast<double>(g.m) / static_cast<double>(g.n) : 0.0;

    ui dmax = 0;
    for (ui v = 0; v < g.n; v++)
        dmax = max(dmax, g.d[v]);
    f.d_max = static_cast<double>(dmax);

    vector<ui> core = compute_core_numbers(g);

    ui max_core = 0;
    for (ui v = 0; v < g.n; v++)
        max_core = max(max_core, core[v]);
    vector<ui> shell_counts(max_core + 1, 0);
    for (ui v = 0; v < g.n; v++)
        shell_counts[core[v]]++;

    vector<ui> sizes;
    ui core_min_val = max_core;
    for (ui c = 0; c <= max_core; c++)
    {
        if (shell_counts[c] > 0)
        {
            sizes.push_back(shell_counts[c]);
            core_min_val = min(core_min_val, c);
        }
    }

    f.core_max = static_cast<double>(max_core);
    f.core_min = g.n > 0 ? static_cast<double>(core_min_val) : 0.0;
    f.shell_num = static_cast<double>(sizes.size());

    double sum = 0, max_s = sizes.empty() ? 0.0 : (double)sizes[0], min_s = sizes.empty() ? 0.0 : (double)sizes[0];
    for (ui s : sizes)
    {
        sum += s;
        max_s = max(max_s, (double)s);
        min_s = min(min_s, (double)s);
    }
    double avg = sizes.empty() ? 0.0 : sum / sizes.size();
    double var = 0;
    for (ui s : sizes)
        var += (s - avg) * (s - avg);
    var = sizes.empty() ? 0.0 : var / sizes.size(); // population std (ddof=0), matching np.std default

    f.shell_avgSize = avg;
    f.shell_maxSize = max_s;
    f.shell_minSize = min_s;
    f.shell_stdSize = sqrt(var);

    return f;
}

/**
 * @brief Loads a booster exported by export_xgb_booster.py and ranks the 24 Scheme 1
 * algorithm labels by predicted probability for one graph instance. The booster file
 * is plain XGBoost format (no sklearn), so this has no Python dependency at runtime.
 */
class Scheme1Predictor
{
public:
    /**
     * @brief Process-wide singleton: the model/meta paths only matter on the first
     * call (the booster is loaded once and reused for every instance this process
     * solves). Fine for this solver's one-model-per-run usage.
     */
    static Scheme1Predictor &instance(const string &model_path, const string &meta_path)
    {
        static Scheme1Predictor inst(model_path, meta_path);
        return inst;
    }

    vector<pair<string, float>> predict_topk(const GraphMLFeatures &feats, int top_k) const
    {
        vector<float> row(feature_cols_.size());
        for (size_t i = 0; i < feature_cols_.size(); i++)
            row[i] = static_cast<float>(feature_value(feats, feature_cols_[i]));

        DMatrixHandle dmat;
        XGB_CHECK(XGDMatrixCreateFromMat(row.data(), 1, row.size(), std::nan(""), &dmat));

        const char *config =
            "{\"type\": 0, \"training\": false, \"iteration_begin\": 0, "
            "\"iteration_end\": 0, \"strict_shape\": false}";
        bst_ulong const *out_shape = nullptr;
        bst_ulong out_dim = 0;
        const float *out_result = nullptr;
        int status = XGBoosterPredictFromDMatrix(booster_, dmat, config, &out_shape, &out_dim, &out_result);
        XGDMatrixFree(dmat);
        if (status != 0)
            throw std::runtime_error(std::string("XGBoost predict error: ") + XGBGetLastError());

        size_t n_out = out_dim > 0 ? static_cast<size_t>(out_shape[out_dim - 1]) : 0;
        size_t n_class = min(classes_.size(), n_out);

        vector<pair<string, float>> ranked;
        ranked.reserve(n_class);
        for (size_t c = 0; c < n_class; c++)
            ranked.push_back({classes_[c], out_result[c]});

        sort(ranked.begin(), ranked.end(), [](const pair<string, float> &a, const pair<string, float> &b)
             { return a.second > b.second; });
        if (static_cast<int>(ranked.size()) > top_k)
            ranked.resize(top_k);
        return ranked;
    }

    /**
     * @brief All classes ranked by predicted probability, descending. Used to build
     * escalation batches in build_ml_race_options (main.cpp), which races the front of
     * this list and pulls in more from it if the race hasn't finished yet.
     */
    vector<pair<string, float>> predict_ranked(const GraphMLFeatures &feats) const
    {
        return predict_topk(feats, static_cast<int>(classes_.size()));
    }

private:
    BoosterHandle booster_ = nullptr;
    vector<string> classes_;
    vector<string> feature_cols_;

    Scheme1Predictor(const string &model_path, const string &meta_path)
    {
        load_meta(meta_path);
        XGB_CHECK(XGBoosterCreate(nullptr, 0, &booster_));
        XGB_CHECK(XGBoosterLoadModel(booster_, model_path.c_str()));
    }

    // model_meta.txt is two lines, written by export_xgb_booster.py:
    //   classes:<label1>,<label2>,...       (label_encoder.classes_ order == model's class index order)
    //   feature_cols:<name1>,<name2>,...    (exact column order the booster was trained on)
    void load_meta(const string &meta_path)
    {
        ifstream in(meta_path);
        if (!in.is_open())
            throw std::runtime_error("Could not open ML model metadata file: " + meta_path +
                                      " (run export_xgb_booster.py first)");
        string line;
        while (getline(in, line))
        {
            size_t colon = line.find(':');
            if (colon == string::npos)
                continue;
            string key = line.substr(0, colon);
            string values = line.substr(colon + 1);
            vector<string> *target = nullptr;
            if (key == "classes")
                target = &classes_;
            else if (key == "feature_cols")
                target = &feature_cols_;
            else
                continue;
            stringstream ss(values);
            string tok;
            while (getline(ss, tok, ','))
                if (!tok.empty())
                    target->push_back(tok);
        }
        if (classes_.empty() || feature_cols_.empty())
            throw std::runtime_error("ML model metadata file missing 'classes' or 'feature_cols': " + meta_path);
    }

    static double feature_value(const GraphMLFeatures &f, const string &name)
    {
        if (name == "k_value")
            return f.k_value;
        if (name == "n")
            return f.n;
        if (name == "m")
            return f.m;
        if (name == "d_avg")
            return f.d_avg;
        if (name == "d_max")
            return f.d_max;
        if (name == "core_max")
            return f.core_max;
        if (name == "core_min")
            return f.core_min;
        if (name == "shell_num")
            return f.shell_num;
        if (name == "shell_avgSize")
            return f.shell_avgSize;
        if (name == "shell_maxSize")
            return f.shell_maxSize;
        if (name == "shell_minSize")
            return f.shell_minSize;
        if (name == "shell_stdSize")
            return f.shell_stdSize;
        throw std::runtime_error("Unknown ML feature column in model_meta.txt: " + name);
    }
};

#endif // USE_ML_PREDICTOR
#endif // MLPREDICTOR_H
