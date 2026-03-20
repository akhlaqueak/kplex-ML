#ifndef REDUCTION_H
#define REDUCTION_H

#include "2th-Reduction.h"
#include "CTPrune.h"
/**
 * @brief second order reduction
 */
class ReductionCTCP: public ReductionBase
{
private:
    ui *triangles;
    Graph *g;
    ui n, m;
    vector<bool> vis; // a cache when counting triangles
    ui *edge_to, *pstart, *deg;
    /**
     * @brief count the number of triangles for each edge and save the results in triangls[]
     * T(n)=O(m + sigma( min(d[from], d[to]) | (from,to) in E) )
     */
    void countTriangles()
    {
        if (triangles == nullptr)
        {
            triangles = new ui[m];
        }
        double start_list = get_system_time_microsecond();
        for (ui i = 0; i < n; i++)
        {
            for (ui j = pstart[i]; j < pstart[i + 1]; j++)
            {
                assert(j < m);
                assert(edge_to[j] < n);
                vis[edge_to[j]] = 1;
            }
            for (ui j = pstart[i]; j < pstart[i + 1]; j++)
            {
                ui v = edge_to[j];
                if (g->d[i] < g->d[v])
                    continue;
                int cnt = 0;
                for (ui a = pstart[v]; a < pstart[v + 1]; a++)
                {
                    ui w = edge_to[a];
                    if (vis[w])
                        cnt++;
                }
                triangles[j] = cnt;
            }
            for (ui j = pstart[i]; j < pstart[i + 1]; j++)
                vis[edge_to[j]] = 0;
        }
        for (ui i = 0; i < n; i++)
        {
            for (ui j = pstart[i]; j < pstart[i + 1]; j++)
            {
                ui v = edge_to[j];
                if (g->d[i] > g->d[v])
                    continue;
                ui ano_edge = lower_bound(edge_to + pstart[v], edge_to + pstart[v + 1], i) - edge_to;
                triangles[j] = triangles[ano_edge];
            }
        }
        list_triangle_time += get_system_time_microsecond() - start_list;
    }
    /**
     * @brief weak reduce
     *
     * @param q containg the vertices that we need to remove; Initially, q is not empty
     * @param vertex_removed a mask to record whether a vertex is in queue
     * @param edge_removed a mask recording whether an edge is removed by strong reduction
     *
     * @return the number of vertices that we remove
     */
    int remove_vertex(queue<ui> &q, vector<bool> &vertex_removed, vector<bool> &edge_removed)
    {
        int cnt = 0;
        while (q.size())
        {
            ui u = q.front();
            q.pop();
            cnt++;
            for (ui i = pstart[u]; i < pstart[u + 1]; i++)
            {
                ui v = edge_to[i];
                if (vertex_removed[v] || edge_removed[i])
                    continue;
                if (--deg[v] + paramK <= lb)
                    q.push(v), vertex_removed[v] = 1;
            }
        }
        return cnt;
    }

public:
    ReductionCTCP(Graph *_g) : g(_g), triangles(nullptr)
    {
        n = g->n;
        m = g->m;
        vis.resize(n, 0);
    }
    ~ReductionCTCP()
    {
        if (triangles != nullptr)
        {
            delete[] triangles;
            triangles = nullptr;
        }
    }
    /**
     * @brief we first remove edges with updating triangles[]; also, we put core-reduce on highest priority.
     * assume we already conduct weak reduce
     */
    void strong_reduce(int lb)
    {
        if(lb+1<=2*paramK) return;
        CTCP(lb);
    }
    void rebuild_graph(vector<bool> &vertex_removed, vector<bool> &edge_removed, ui *que = nullptr)
    {
        ui *q = que;
        if (q == nullptr)
        {
            q = new ui[n];
        }
        ui new_n = 0;
        assert(n == g->map_refresh_id.size());
        vector<ui> new_map(n);
        for (ui i = 0; i < n; i++)
            if (!vertex_removed[i])
            {
                new_map[new_n] = g->map_refresh_id[i];
                q[i] = new_n++;
            }
        new_map.resize(new_n);
        g->map_refresh_id = new_map;
        ui *new_pstart = new ui[new_n + 1];
        ui *new_d = new ui[new_n];
        ui j = 0;
        for (ui i = 0; i < n; i++)
        {
            if (vertex_removed[i])
                continue;
            ui u = q[i];
            new_pstart[u] = j;
            for (ui p = pstart[i]; p < pstart[i + 1]; p++)
            {
                ui v = edge_to[p];
                if (!vertex_removed[v] && !edge_removed[p])
                    edge_to[j++] = q[edge_to[p]];
            }
            new_d[u] = j - new_pstart[u];
        }
        new_pstart[new_n] = j;
        delete[] g->d;
        delete[] g->pstart;
        g->d = new_d;
        g->pstart = new_pstart;
        deg = new_d;
        pstart = new_pstart;
        if (j * 2 < m)
        {
            ui *new_edge_to = new ui[j];
            memcpy(new_edge_to, edge_to, sizeof(ui) * j);
            delete[] g->edge_to;
            g->edge_to = new_edge_to;
            edge_to = new_edge_to;

            if (triangles != nullptr)
                delete[] triangles;
            triangles = new ui[j];
        }
        g->m = m = j;
        g->n = n = new_n;
        if (que == nullptr)
        {
            delete[] q;
        }
    }
    /**
     * @brief core-truss co-pruning
     */
    void CTCP(int lb)
{
    // =========================
    // Bind graph data (same as before)
    // =========================
    edge_to = g->edge_to;
    pstart = g->pstart;
    deg = g->d;

    ui n_local = n;
    ui m_local = m;

    // =========================
    // Allocate CTPrune buffers
    // =========================
    ui *peel_sequence = new ui[n_local];
    ui *out_mapping   = new ui[n_local];
    ui *rid           = new ui[n_local];

    // initialize
    for (ui i = 0; i < n_local; i++) {
        peel_sequence[i] = i;
        out_mapping[i]   = g->map_refresh_id[i];
    }

    // =========================
    // Call CTPrune
    // =========================
    // We reinterpret lb as triangle threshold
    CTPrune::truss_pruning(
        n_local,
        m_local,
        lb,
        peel_sequence,
        out_mapping,
        rid,
        pstart,
        edge_to,
        deg,
        false
    );

    // =========================
    // Update graph in-place
    // =========================
    n = g->n = n_local;
    m = g->m = m_local;

    g->pstart = pstart;
    g->edge_to = edge_to;
    g->d = deg;

    // update mapping (important)
    g->map_refresh_id.assign(out_mapping, out_mapping + n);

    // =========================
    // Cleanup
    // =========================
    delete[] peel_sequence;
    delete[] out_mapping;
    delete[] rid;
}    
};

class Reduction{
    ReductionBase* reduce;
    public:
    Reduction(Graph *_g){
        if(PARA.ctcp=="CTCP")
            reduce = new ReductionCTCP(_g);
        else
            reduce = new ReductionCFCTCP(_g);
    }
     void strong_reduce(int lb){
        reduce->strong_reduce(lb);
    }
};
#endif