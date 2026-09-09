#ifndef REDUCTION_CFCTCP_H
#define REDUCTION_CFCTCP_H
#include "CTPrune.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <queue>
class ReductionBase
{
public:
    virtual ~ReductionBase() = default;
    virtual void strong_reduce(int lb) = 0;
};
/**
 * @brief second order reduction: CF-CTCP
 */
class ReductionCFCTCP : public ReductionBase
{
private:
    // the edge in the edge queue is stored as {from_v, edge_id, remove_time}
    struct EdgeForQ_E
    {
        ui u, edge_id;
        ui remove_time;
    };
    // my queue
    class Queue
    {
        ui *q;
        ui hh, tt;

    public:
        Queue(ui max_size) : q(nullptr), hh(1), tt(0)
        {
            q = new ui[max_size + 1];
        }
        ~Queue()
        {
            if (q != nullptr)
            {
                delete[] q;
            }
        }
        void push(ui u)
        {
            q[++tt] = u;
        }
        void pop()
        {
            hh++;
        }
        ui front()
        {
            return q[hh];
        }
        int size()
        {
            return tt + 1 - hh;
        }
    };

    // G-fast: only vertices in G-fast are useful
    Graph &G_fast;
    ui *triangles, *compute_time, *another_edge;
    vector<char> vertex_removed_from_G_fast, edge_removed_from_G_fast;
    ui n, m;
    ui timestamp;
    // G-slow: this is used to update triangle count
    vector<char> edge_removed_from_G_slow;
    // cache
    vector<char> vis;

    /**
     * @brief reduce G-fast to (lb+1-k)-core
     *
     * @param q_v the vertices that must remove
     * @param q_e the edges removed should be pushed to queue
     */
    void core_reduce(queue<ui> &q_v, int lb, vector<EdgeForQ_E> &q_e)
    {
        ui *pstart = G_fast.pstart, *edge_to = G_fast.edge_to;
        ui *d = G_fast.d;
        while (q_v.size())
        {
            ui u = q_v.front();
            q_v.pop();
            for (ui i = pstart[u]; i < pstart[u + 1]; i++)
            {
                if (edge_removed_from_G_fast[i])
                    continue;
                ui v = edge_to[i];
                if (!vertex_removed_from_G_fast[v])
                {
                    d[v]--;
                    if (d[v] + paramK == lb)
                    {
                        q_v.push(v);
                        vertex_removed_from_G_fast[v] = 1;
                    }
                    else
                        q_e.push_back({u, i, timestamp});
                }
            }
        }
    }

    /**
     * @brief remove the edge(u,v) and reduce G-fast to (lb+1-k)-core immediately
     *
     * @param u from
     * @param v to
     * @param edge_id make sure that edge_to[edge_id] = v
     * @param q_e the edges removed should be pushed to queue
     */
    void remove_edge_from_G_fast(ui u, ui v, ui edge_id, int lb, vector<EdgeForQ_E> &q_e)
    {
        q_e.push_back({u, edge_id, ++timestamp});
        edge_removed_from_G_fast[edge_id] = 1;
        ui another_edge_id = another_edge[edge_id];
        edge_removed_from_G_fast[another_edge_id] = 1;
        queue<ui> q;
        if (--G_fast.d[u] + paramK == lb)
        {
            q.push(u);
            vertex_removed_from_G_fast[u] = 1;
        }
        if (--G_fast.d[v] + paramK == lb)
        {
            q.push(v);
            vertex_removed_from_G_fast[v] = 1;
        }
        if (q.size())
        {
            core_reduce(q, lb, q_e);
        }
    }

    /**
     * @brief reduce G-fast to (lb+1-k)-core for the first round: we don't consider updating triangles count
     *
     * @param q_v the vertices that must remove
     *
     * @return the number of vertices we remove
     */
    ui core_reduce(Queue &q_v, int lb)
    {
        ui *pstart = G_fast.pstart, *edge_to = G_fast.edge_to;
        ui *d = G_fast.d;
        ui cnt = 0;
        while (q_v.size())
        {
            ui u = q_v.front();
            q_v.pop();
            cnt++;
            for (ui i = pstart[u]; i < pstart[u + 1]; i++)
            {
                if (edge_removed_from_G_fast[i])
                    continue;
                ui v = edge_to[i];
                if (!vertex_removed_from_G_fast[v])
                {
                    d[v]--;
                    if (d[v] + paramK == lb)
                    {
                        q_v.push(v);
                        vertex_removed_from_G_fast[v] = 1;
                    }
                }
            }
        }
        return cnt;
    }

    /**
     * @brief refresh the graph which can fast the computation of counting triangles
     *
     * @param u the next vertex that we are going to enumerate its edges
     *
     * @return the index of u in the new graph(if u is not in new graph, return the next vertex of u)
     */
    ui rebuild_graph_for_first_round(ui u)
    {
        ui *pstart = G_fast.pstart, *edge_to = G_fast.edge_to;
        ui *d = G_fast.d;
        ui *id_map = new ui[n];
        ui new_n = 0;
        vector<ui> map_refresh_id(n);
        for (ui i = 0; i < n; i++)
        {
            if (!vertex_removed_from_G_fast[i])
            {
                map_refresh_id[new_n] = G_fast.map_refresh_id[i];
                id_map[i] = new_n++;
            }
        }
        ui new_idx_u = new_n;
        for (ui i = u; i < n; i++)
            if (!vertex_removed_from_G_fast[i])
            {
                new_idx_u = id_map[i];
                break;
            }
        map_refresh_id.resize(new_n);
        G_fast.map_refresh_id = map_refresh_id;
        ui new_m = 0;
        ui *new_pstart = new ui[new_n + 1];
        for (ui u = 0; u < n; u++)
        {
            if (vertex_removed_from_G_fast[u])
                continue;
            new_pstart[id_map[u]] = new_m;
            for (ui i = pstart[u]; i < pstart[u + 1]; i++)
            {
                if (edge_removed_from_G_fast[i])
                    continue;
                ui v = edge_to[i];
                if (vertex_removed_from_G_fast[v])
                    continue;
                edge_to[new_m++] = id_map[v];
            }
        }
        new_pstart[new_n] = new_m;
        ui *new_d = new ui[new_n];
        for (ui i = 0; i < new_n; i++)
            new_d[i] = new_pstart[i + 1] - new_pstart[i];
        delete[] pstart;
        delete[] d;
        delete[] id_map;
        G_fast.pstart = new_pstart;
        G_fast.d = new_d;
        n = G_fast.n = new_n;
        m = G_fast.m = new_m;
        edge_removed_from_G_fast.clear();
        edge_removed_from_G_fast.resize(m, 0);
        vertex_removed_from_G_fast.clear();
        vertex_removed_from_G_fast.resize(n, 0);
        vis.resize(n, 0);
        return new_idx_u;
    }

    /**
     * @brief for the first round, we think almost each edge can be removed
     * so we compute a near-exact triangle count(slightly greater than the fact) and judge if it can be removed
     *
     * @param q_v a queue that can be shared
     * @param start_u for v<start_u, we don't consider the edges related to v
     */
    void first_round_reduce(int lb, Queue &q_v, ui start_u = 0)
    {
        ui *pstart = G_fast.pstart, *edge_to = G_fast.edge_to;
        ui *d = G_fast.d;
        Timer t;
        ui remove_vertex_cnt = 0;
        for (ui u = start_u; u < n; u++)
        {
            if (vertex_removed_from_G_fast[u])
                continue;
            if (d[u] + 1 == n) // u is connected to all vertices
                continue;
            // cache: record the neighbor of u(like a hash operation)
            for (ui i = pstart[u]; i < pstart[u + 1]; i++)
            {
                if (edge_removed_from_G_fast[i])
                    continue;
                ui v = edge_to[i];
                if (vertex_removed_from_G_fast[v])
                    continue;
                vis[v] = 1;
            }
            // enmerate the neighbor of u, and each edge we compute only once
            for (ui i = pstart[u]; i < pstart[u + 1]; i++)
            {
                if (edge_removed_from_G_fast[i])
                    continue;
                ui v = edge_to[i];
                if (vertex_removed_from_G_fast[v])
                    continue;
                if (v >= u) // make sure v<u in degeneracy order
                    break;
                ui edge_cnt = 0;
                // compute the triangle count of (u,v)
                for (ui j = pstart[v]; j < pstart[v + 1]; j++)
                {
                    if (edge_removed_from_G_fast[j])
                        continue;
                    ui w = edge_to[j];
                    if (vertex_removed_from_G_fast[w])
                        continue;
                    if (vis[w]) // w is a common neighbor
                        edge_cnt++;
                }
                if (edge_cnt + 2 * paramK <= lb) // remove (u,v)
                {
                    edge_removed_from_G_fast[i] = true;
                    ui another_edge_id = find(edge_to + pstart[v], edge_to + pstart[v + 1], u) + pstart[v];
                    edge_removed_from_G_fast[another_edge_id] = true;
                    if (--d[u] + paramK == lb)
                    {
                        q_v.push(u);
                        vertex_removed_from_G_fast[u] = true;
                    }
                    if (--d[v] + paramK == lb)
                    {
                        q_v.push(v);
                        vertex_removed_from_G_fast[v] = true;
                    }
                    if (q_v.size())
                    {
                        remove_vertex_cnt += core_reduce(q_v, lb);
                        if (vertex_removed_from_G_fast[u])
                            break;
                    }
                    // note that v is not a neighbor of u now
                    vis[v] = 0;
                }
            }
            // clear the cache
            for (ui i = pstart[u]; i < pstart[u + 1]; i++)
            {
                ui v = edge_to[i];
                vis[v] = 0;
            }
            // shrink current graph to boost the computation
            if (remove_vertex_cnt * 4 >= n)
            {
                ui new_start_u = rebuild_graph_for_first_round(u + 1);
                list_triangle_time += t.get_time();
                first_round_reduce(lb, q_v, new_start_u);
                return;
            }
        }
        rebuild_graph_for_first_round(0);
        list_triangle_time += t.get_time();
    }

public:
    ReductionCFCTCP(Graph *_g) : G_fast(*_g), triangles(nullptr), compute_time(nullptr), another_edge(nullptr),
                                 timestamp(0)
    {
        n = G_fast.n;
        m = G_fast.m;
        vis.resize(n, 0);
    }
    ~ReductionCFCTCP()
    {
        vector<ui *> ptrs{triangles, compute_time, another_edge};
        for (auto p : ptrs)
            if (p != nullptr)
            {
                delete[] p;
            }
    }
    /**
     * the entrance of our second-order reduction
     */
    void strong_reduce(int lb)
    {
        // CF-CTCP uses CTPrune's forward-oriented enumeration for its initial
        // edge supports. The subsequent dynamic maintenance remains unchanged.
        vertex_removed_from_G_fast.assign(n, 0);
        edge_removed_from_G_fast.assign(m, 0);
        ui *pstart = G_fast.pstart, *edge_to = G_fast.edge_to;
        triangles = new ui[m];
        compute_time = new ui[m];
        another_edge = new ui[m];
        edge_removed_from_G_slow.assign(m, 0);
        CF_CTCP(lb);
    }
    void rebuild_graph()
    {
        ui *pstart = G_fast.pstart, *edge_to = G_fast.edge_to;
        ui *d = G_fast.d;
        ui *id_map = new ui[n];
        ui new_n = 0;
        vector<ui> map_refresh_id(n);
        for (ui i = 0; i < n; i++)
        {
            if (!vertex_removed_from_G_fast[i])
            {
                map_refresh_id[new_n] = G_fast.map_refresh_id[i];
                id_map[i] = new_n++;
            }
        }
        map_refresh_id.resize(new_n);
        G_fast.map_refresh_id = map_refresh_id;
        ui new_m = 0;
        ui *new_pstart = new ui[new_n + 1];
        for (ui u = 0; u < n; u++)
        {
            if (vertex_removed_from_G_fast[u])
                continue;
            new_pstart[id_map[u]] = new_m;
            for (ui i = pstart[u]; i < pstart[u + 1]; i++)
            {
                if (edge_removed_from_G_fast[i])
                    continue;
                ui v = edge_to[i];
                if (vertex_removed_from_G_fast[v])
                    continue;
                edge_to[new_m++] = id_map[v];
            }
        }
        new_pstart[new_n] = new_m;
        ui *new_d = new ui[new_n];
        for (ui i = 0; i < new_n; i++)
            new_d[i] = new_pstart[i + 1] - new_pstart[i];
        delete[] pstart;
        delete[] d;
        delete[] id_map;
        G_fast.pstart = new_pstart;
        G_fast.d = new_d;
        n = G_fast.n = new_n;
        m = G_fast.m = new_m;
        for (ui u = 0; u < n; u++)
        {
            for (ui i = G_fast.pstart[u]; i < G_fast.pstart[u + 1]; i++)
            {
                ui v = G_fast.edge_to[i];
            }
        }
    }
    void initialize_triangles_oriented()
    {
        ui *pstart = G_fast.pstart, *edge_to = G_fast.edge_to;
        vector<ui> peel_sequence(n), rank(n), pend(n + 1), pend_buffer(n + 1);

        // G_fast is already degeneracy-recoded, so its vertex IDs are the peel
        // sequence CTPrune expects. Pack forward edges without recomputing it.
        for (ui u = 0; u < n; u++)
            peel_sequence[u] = u;
        CTPrune::orient_graph(n, m, peel_sequence.data(), pstart, pend.data(), edge_to, rank.data());
        CTPrune::oriented_triangle_counting(n, m, pstart, pend.data(), edge_to, triangles, rank.data());
        CTPrune::reorganize_oriented_graph(n, triangles, pstart, pend.data(), pend_buffer.data(), edge_to, another_edge, rank.data());

        for (ui u = 0; u < n; u++)
            G_fast.d[u] = pstart[u + 1] - pstart[u];

        // CF-CTCP's later stage expects a timestamp on every CSR copy.
        ui initial_time = ++timestamp;
        fill(compute_time, compute_time + m, initial_time);
    }
    void CF_CTCP(int lb)
    {
        vector<EdgeForQ_E> q_e;
        ui *pstart = G_fast.pstart, *edge_to = G_fast.edge_to;
        Timer t;
        initialize_triangles_oriented();

        // Seed CF-CTCP's existing dynamic maintenance with initially invalid
        // edges. Each undirected edge is examined once through its backward CSR
        // copy; all later propagation below is unchanged.
        for (ui u = 0; u < n; u++)
        {
            if (vertex_removed_from_G_fast[u])
                continue;
            for (ui i = pstart[u]; i < pstart[u + 1]; i++)
            {
                if (edge_removed_from_G_fast[i])
                    continue;
                ui v = edge_to[i];
                if (vertex_removed_from_G_fast[v])
                    continue;
                vis[v] = 1;
            }
            // enmerate the neighbor of u, and each edge we compute only once
            for (ui i = pstart[u]; i < pstart[u + 1]; i++)
            {
                if (edge_removed_from_G_fast[i])
                    continue;
                ui v = edge_to[i];
                if (vertex_removed_from_G_fast[v])
                    continue;
                if (v >= u)
                    break;
                if (triangles[i] + 2 * paramK <= lb)
                {
                    remove_edge_from_G_fast(u, v, i, lb, q_e);
                    if (vertex_removed_from_G_fast[u])
                        break;
                }
            }
        }
        list_triangle_time += t.get_time();
        if (!q_e.size())
            return;
        // update the information of triangle count exactly
        ui front_idx = 0;
        while (front_idx < q_e.size())
        {
            // auto h = q_e.front(); q_e.pop();
            auto &h = q_e[front_idx++];
            ui u = h.u, edge_id = h.edge_id, v = edge_to[edge_id], time = h.remove_time;
            edge_removed_from_G_slow[edge_id] = 1;
            ui another_edge_id = another_edge[edge_id];
            edge_removed_from_G_slow[another_edge_id] = 1;
            if (vertex_removed_from_G_fast[u] && vertex_removed_from_G_fast[v])
                continue;
            if (!vertex_removed_from_G_fast[u])
            {
                // i for G-fast and j for G-slow
                for (ui i = pstart[u], j = pstart[v]; i < pstart[u + 1] && j < pstart[v + 1];)
                {
                    if (edge_removed_from_G_fast[i] || vertex_removed_from_G_fast[edge_to[i]])
                    {
                        i++;
                        continue;
                    }
                    if (edge_removed_from_G_slow[j])
                    {
                        j++;
                        continue;
                    }
                    if (edge_to[i] < edge_to[j])
                    {
                        i++;
                        continue;
                    }
                    else if (edge_to[i] > edge_to[j])
                    {
                        j++;
                        continue;
                    }
                    else // now we find a w in N_{G_f}(u) \cap N_{G_s}(v)
                    {
                        if (compute_time[i] > time) // the removing of (u,v) will not affect (u,w) because \Delta(u,w) is computed after removing (u,v)
                        {
                            i++;
                            j++;
                            continue;
                        }
                        ui w = edge_to[i];
                        if (--triangles[i] + 2 * paramK <= lb)
                        {
                            remove_edge_from_G_fast(u, w, i, lb, q_e);
                            if (vertex_removed_from_G_fast[u])
                                break;
                        }
                        else
                        {
                            ui another_edge_id = another_edge[i];
                            triangles[another_edge_id]--;
                        }
                        i++;
                        j++;
                    }
                }
            }

            if (!vertex_removed_from_G_fast[v])
            {
                // i for G-fast and j for G-slow
                for (ui i = pstart[v], j = pstart[u]; i < pstart[v + 1] && j < pstart[u + 1];)
                {
                    if (edge_removed_from_G_fast[i] || vertex_removed_from_G_fast[edge_to[i]])
                    {
                        i++;
                        continue;
                    }
                    if (edge_removed_from_G_slow[j])
                    {
                        j++;
                        continue;
                    }
                    if (edge_to[i] < edge_to[j])
                    {
                        i++;
                        continue;
                    }
                    else if (edge_to[i] > edge_to[j])
                    {
                        j++;
                        continue;
                    }
                    else // now we find a w in N_{G_f}(v) \cap N_{G_s}(u)
                    {
                        if (compute_time[i] >= time) // the removing of (u,v) will not affect (v,w) because \Delta(v,w) is computed after removing (u,v)
                        {
                            i++;
                            j++;
                            continue;
                        }
                        ui w = edge_to[i];
                        if (--triangles[i] + 2 * paramK <= lb)
                        {
                            remove_edge_from_G_fast(v, w, i, lb, q_e);
                            if (vertex_removed_from_G_fast[v])
                                break;
                        }
                        else
                        {
                            ui another_edge_id = another_edge[i];
                            triangles[another_edge_id]--;
                        }
                        i++;
                        j++;
                    }
                }
            }
        }
        rebuild_graph();
    }
};

/**
 * @brief second order reduction
 */
class ReductionCTCP: public ReductionBase
{
private:
    ui *triangles;
    Graph *g;
    ui n, m;
    vector<char> vis; // a cache when counting triangles
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
    int remove_vertex(queue<ui> &q, vector<char> &vertex_removed, vector<char> &edge_removed)
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
    void rebuild_graph(vector<char> &vertex_removed, vector<char> &edge_removed, ui *que = nullptr)
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
        edge_to = g->edge_to;
        pstart  = g->pstart;
        deg     = g->d;

    
        int degree_threshold   = lb - paramK + 1;
        int triangle_threshold = lb - 2 * paramK + 1;
    
        if (triangle_threshold <= 0 || degree_threshold <= 0)
            return;
        // vector<ui> psq = g->degenerate();
        // ui *peel_sequence = psq.data();
        ui *rid           = new ui[n];
        ui *psq           = new ui[n];
        g->degenerate(psq);

        CTPrune::core_truss_copruning(
            g->n,
            g->m,
            (ui)degree_threshold,
            (ui)triangle_threshold,
            psq,
            g->map_refresh_id.data(),
            rid,
            g->pstart,
            g->edge_to,
            g->d,
            false
        );
        g->map_refresh_id.resize(g->n);
        
        delete[] rid;
        delete[] psq;

    } 
};

class Reduction{
    ReductionBase* reduce;
    Graph *g;
    public:
    Reduction(Graph *_g){
        g=_g;
        if(PARA.ctcp=="CTCP")
            reduce = new ReductionCTCP(_g);
        else
            reduce = new ReductionCFCTCP(_g);
    }
    ~Reduction(){
        delete reduce;
    }
    void strong_reduce(int lb){
        if(lb >= g->n) return;
        reduce->strong_reduce(lb);
    }
};
#endif
