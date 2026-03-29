#ifndef _KPLEX_BB_MATRIX_
#define _KPLEX_BB_MATRIX_

#include "Utility.h"
#include "Graph.h"
// #define _SECOND_ORDER_PRUNING_
ui altrb_cnt = 0;
ll dfs_cnt = 0;

#define CSIZE (R_end - S_end)
class KPLEX_BB_MATRIX
{
	using Set = MyBitset;

private:
	long long n;
	vector<int> ids;
	char *matrix;
	long long matrix_size;
	set<int> solution;
	ui root_v;
#ifdef _SECOND_ORDER_PRUNING_
	ui *cn;
	std::queue<std::pair<ui, ui>> Qe;
	std::vector<std::pair<ui, ui>> removed_edges;
	long long removed_edges_n;
#endif

	ui *degree;
	ui *degree_in_S;

	ui K;
	ui *best_solution;
	ui lb;
	ui _UB_;

	ui *neighbors;
	ui *nonneighbors;
	ui *S2;

	ui *SR;		// union of S and R, where S is at the front
	ui *SR_rid; // reverse ID for SR
	std::queue<ui> Qv;
	ui *level_id;
	ui max_level;

	std::vector<std::pair<ui, ui>> vp;
	std::vector<ui> non_adj;

	// std::vector<std::pair<ui,ui> > vp2;

	bool sparse;
	vector<ui> B, PI, PIMax, ISc, peelOrder, psz;
	ui *LPI;
	MBitSet bmp;
	ui sz1h;
	bool found_larger = false;
	bool ctcp_enabled = false;
	bool all_kplex_search, forward_sol;

	// AltRB Stuff
	// info of the search tree
	vector<int> loss_cnt; // loss_cnt[v] = |S| - |N(v) \cap S|, i.e., non-neighbors of v in S

	Graph_adjacent *ptr_g;

public:
	ui best_n_edges;
	vector<vector<ui>> all_kplexes;
	ui nmkp = 0;
	KPLEX_BB_MATRIX(ui K, bool _ds = false)
	{
		all_kplex_search = _ds;
		this->K = K;
		n = 0;
		matrix = nullptr;
		matrix_size = 0;
		forward_sol = false;
#ifdef _SECOND_ORDER_PRUNING_
		cn = nullptr;
		removed_edges_n = 0;
#endif

		degree = degree_in_S = nullptr;

		best_solution = nullptr;
		lb = _UB_ = 0;

		neighbors = nonneighbors = nullptr;
		S2 = nullptr;

		SR = SR_rid = nullptr;
		level_id = nullptr;
		max_level = 0;
		best_n_edges = 0;
	}

	~KPLEX_BB_MATRIX()
	{
		if (matrix != NULL)
		{
			delete[] matrix;
			matrix = NULL;
		}
#ifdef _SECOND_ORDER_PRUNING_
		if (cn != NULL)
		{
			delete[] cn;
			cn = NULL;
		}
#endif
		if (degree != NULL)
		{
			delete[] degree;
			degree = NULL;
		}
		if (degree_in_S != NULL)
		{
			delete[] degree_in_S;
			degree_in_S = NULL;
		}

		if (best_solution != NULL)
		{
			delete[] best_solution;
			best_solution = NULL;
		}
		if (SR != NULL)
		{
			delete[] SR;
			SR = NULL;
		}
		if (SR_rid != NULL)
		{
			delete[] SR_rid;
			SR_rid = NULL;
		}
		if (neighbors != NULL)
		{
			delete[] neighbors;
			neighbors = NULL;
		}
		if (nonneighbors != NULL)
		{
			delete[] nonneighbors;
			nonneighbors = NULL;
		}
		if (level_id != NULL)
		{
			delete[] level_id;
			level_id = NULL;
		}
	}

	void allocateMemory(ui n)
	{
		if (n <= 0)
			return;

		matrix_size = 1;
		// printf("matrix size: %lldMB\n", (((long long)UB)*v_n*4)/1024/1024);
		matrix = new char[matrix_size];
#ifdef _SECOND_ORDER_PRUNING_
		cn = new ui[matrix_size];
#endif

		degree = new ui[n];
		degree_in_S = new ui[n];
		best_solution = new ui[n];
		SR = new ui[n];
		SR_rid = new ui[n];
		neighbors = new ui[n];
		nonneighbors = new ui[n];
		S2 = new ui[n];
		level_id = new ui[n];
		B.reserve(n);
		LPI = new ui[matrix_size];

		peelOrder.resize(n);
		psz.resize(n);
		PI.reserve(n);
		PIMax.reserve(n);
		ISc.reserve(n);
		bmp.init(n);
	}

	void load_graph(Graph_adjacent &g)
	{
		ptr_g = &g;
		ids = g.vertex_id;
		n = g.size();
		loss_cnt.resize(n);

		if (((long long)n) * n > matrix_size)
		{
			do
			{
				matrix_size *= 2;
			} while (((long long)n) * n > matrix_size);
			delete[] matrix;
			matrix = new char[matrix_size];
			delete[] LPI;
			LPI = new ui[matrix_size];
#ifdef _SECOND_ORDER_PRUNING_
			delete[] cn;
			cn = new ui[matrix_size];
#endif
		}

#ifdef _SECOND_ORDER_PRUNING_
		memset(cn, 0, sizeof(ui) * ((long long)n) * n);
#endif
		double ne = 0;
		memset(matrix, 0, sizeof(char) * ((long long)n) * n);

		for (ui u = 0; u < g.size(); u++)
		{
			auto &adj_u = g.adj_matrix[u];
			degree[u] = adj_u.size();
			ne += degree[u];
			for (auto v : adj_u)
				matrix[u * n + v] = 1;
		}

		sparse = ne / n / (n - 1) < 0.90;

#ifndef NDEBUG
		// printf("load graph of size n=%lld, m=%lu\n", n, vp.size());
		// for(ui i = 0;i < vp.size();i ++) printf("%d %d\n", vp[i].first, vp[i].second);
#endif
	}

	void kPlex(ui lb, ui _UB_, set<int> &solution, ui root_v)
	{
		this->lb = lb;
		this->_UB_ = _UB_;
		this->root_v = root_v;
		if (K <= 1)
		{
			printf("For the special case of computing maximum clique, please invoke SOTA maximum clique solver!\n");
			return;
		}
		ui n_edges = best_n_edges;
		ui R_end;
		initialization(R_end, false);
		if (R_end && lb < _UB_)
			BB_search(0, R_end, 1);
		if (this->solution.size() > solution.size())
			solution = this->solution;
	}

private:
	void readGraph_binary(char *dir)
	{
		FILE *f = Utility::open_file((std::string(dir) + std::string("/b_degree.bin")).c_str(), "rb");

		int tt;
		fread(&tt, sizeof(int), 1, f);
		if (tt != sizeof(int))
		{
			printf("sizeof int is different: edge.bin(%d), machine(%d)\n", tt, (int)sizeof(int));
			return;
		}
		ui m;
		fread(&n, sizeof(int), 1, f);
		fread(&m, sizeof(int), 1, f);
		printf("n = %u, m = %u\n", n, m / 2);

		allocateMemory(n);

		if (((long long)n) * n > matrix_size)
		{
			do
			{
				matrix_size *= 2;
			} while (((long long)n) * n > matrix_size);
			delete[] matrix;
			matrix = new char[matrix_size];
#ifdef _SECOND_ORDER_PRUNING_
			delete[] cn;
			cn = new ui[matrix_size];
#endif
		}

		fread(degree, sizeof(ui), n, f);
		fclose(f);

		f = Utility::open_file((std::string(dir) + std::string("/b_adj.bin")).c_str(), "rb");

#ifdef _SECOND_ORDER_PRUNING_
		memset(cn, 0, sizeof(ui) * n * n);
#endif
		memset(matrix, 0, sizeof(char) * n * n);
		for (ui i = 0; i < n; i++)
			if (degree[i] > 0)
			{
				fread(neighbors, sizeof(ui), degree[i], f);
				for (ui j = 0; j < degree[i]; j++)
					matrix[i * n + neighbors[j]] = 1;
				ui d = 0;
				for (ui j = 0; j < n; j++)
					if (matrix[i * n + j])
						++d;
				if (d != degree[i])
				{
					printf("%u may have duplicate edges\n", i);
					degree[i] = d;
				}
			}
		fclose(f);
	}

	void initialization(ui &R_end, bool must_include_0)
	{
		// the following computes a degeneracy ordering and a heuristic solution
		ui *peel_sequence = neighbors;
		ui *core = nonneighbors;
		ui *vis = SR;
		memset(vis, 0, sizeof(ui) * n);
		ui max_core = 0, idx = n;
		for (ui i = 0; i < n; i++)
		{
			ui u, min_degree = n;
			for (ui j = 0; j < n; j++)
				if (!vis[j] && degree[j] < min_degree)
				{
					u = j;
					min_degree = degree[j];
				}
			if (min_degree > max_core)
				max_core = min_degree;
			core[u] = max_core;
			peel_sequence[i] = u;
			peelOrder[u] = i;
			vis[u] = 1;

			if (idx == n && min_degree + K >= n - i)
				idx = i;

			for (ui j = 0; j < n; j++)
				if (!vis[j] && matrix[u * n + j])
					--degree[j];
		}
		if (!all_kplex_search && (n - idx > lb))
		{
			lb = n - idx;
			solution.clear();
			for (ui i = idx; i < n; i++)
				solution.insert(ids[peel_sequence[i]]);
			printf("Degen find a solution of size %u\n", lb);
		}

		R_end = 0;
		for (ui i = 0; i < n; i++)
			SR_rid[i] = n;
		for (ui i = 0; i < n; i++)
			if (core[i] + K > lb)
			{
				SR[R_end] = i;
				SR_rid[i] = R_end;
				++R_end;
			}

		if ((must_include_0 && SR_rid[0] == n) || lb >= _UB_)
		{
			R_end = 0;
			return;
		}

		for (ui i = 0; i < R_end; i++)
		{
			ui u = SR[i];
			degree[u] = degree_in_S[u] = 0;
			for (ui j = 0; j < R_end; j++)
				if (matrix[u * n + SR[j]])
					++degree[u];
		}

		memset(level_id, 0, sizeof(ui) * n);
		for (ui i = 0; i < R_end; i++)
			level_id[SR[i]] = n;

		if (!Qv.empty())
			printf("!!! Something wrong. Qv must be empty in initialization\n");

#ifdef _SECOND_ORDER_PRUNING_
		for (ui i = 0; i < R_end; i++)
		{
			ui neighbors_n = 0;
			char *t_matrix = matrix + SR[i] * n;
			for (ui j = 0; j < R_end; j++)
				if (t_matrix[SR[j]])
					neighbors[neighbors_n++] = SR[j];
			for (ui j = 0; j < neighbors_n; j++)
				for (ui k = j + 1; k < neighbors_n; k++)
				{
					++cn[neighbors[j] * n + neighbors[k]];
					++cn[neighbors[k] * n + neighbors[j]];
				}
		}

		while (!Qe.empty())
			Qe.pop();
		for (ui i = 0; i < R_end; i++)
			for (ui j = i + 1; j < R_end; j++)
			{
				if (matrix[SR[i] * n + SR[j]] && upper_bound_based_prune(0, SR[i], SR[j]))
				{
					Qe.push(std::make_pair(SR[i], SR[j]));
				}
			}
		removed_edges_n = 0;
#endif

		if (!remove_vertices_and_edges_with_prune(0, R_end, 0))
			R_end = 0;
	}

	void store_solution(ui size)
	{
		if (size <= lb)
		{
			printf("!!! the solution to store is no larger than the current best solution!");
			return;
		}
		ui n_edges = 0;
		for (ui i = 0; i < size; i++)
		{
			if (forward_sol)
				n_edges += degree[SR[i]];
			else
				n_edges += degree_in_S[SR[i]];
		}
		forward_sol = false;
		Set S(n);
		for (ui i = 0; i < size; i++)
		{
			S.set(SR[i]);
		}

		update_lb(S);

		// printf("!!! BB_Search found a kplex of size: %u, n_edges: %u \n", size, n_edges);
		// lb = size;
		// found_larger = true;
		best_n_edges = n_edges;
	}

	bool is_kplex(ui R_end)
	{
		for (ui i = 0; i < R_end; i++)
			if (degree[SR[i]] + K < R_end)
				return false;
		forward_sol = true;
		return true;
	}

	void BB_search(ui S_end, ui R_end, ui level)
	{
		if (S_end > lb)
			store_solution(S_end);
		if (R_end > lb && is_kplex(R_end))
			store_solution(R_end);
		if (R_end <= lb + 1 || lb >= _UB_)
			return;
		dfs_cnt++;
#ifndef NDEBUG
		for (ui i = 0; i < R_end; i++)
		{
			ui d1 = 0, d2 = 0;
			for (ui j = 0; j < S_end; j++)
				if (matrix[SR[i] * n + SR[j]])
					++d1;
			d2 = d1;
			for (ui j = S_end; j < R_end; j++)
				if (matrix[SR[i] * n + SR[j]])
					++d2;
			assert(d1 == degree_in_S[SR[i]]);
			assert(d2 == degree[SR[i]]);
		}
		for (ui i = 0; i < S_end; i++)
			assert(degree_in_S[SR[i]] + K >= S_end);
		for (ui i = S_end; i < R_end; i++)
			assert(degree_in_S[SR[i]] + K > S_end);
		for (ui i = 0; i < S_end; i++)
			if (degree_in_S[SR[i]] + K == S_end)
			{
				char *t_matrix = matrix + SR[i] * n;
				for (ui j = S_end; j < R_end; j++)
					assert(t_matrix[SR[j]]);
			}
		for (ui i = 0; i < R_end; i++)
			assert(level_id[SR[i]] > level);
#ifdef _SECOND_ORDER_PRUNING_
		for (ui i = 0; i < R_end; i++)
			for (ui j = i + 1; j < R_end; j++)
			{
				ui v = SR[i], w = SR[j];
				ui common_neighbors = 0;
				for (ui k = S_end; k < R_end; k++)
					if (matrix[SR[k] * n + v] && matrix[SR[k] * n + w])
						++common_neighbors;
				assert(cn[v * n + w] == common_neighbors);
				assert(cn[w * n + v] == common_neighbors);
			}
#endif
#endif

		ui old_removed_edges_n = 0, old_S_end = S_end, old_R_end = R_end;
		assert(Qv.empty());
#ifdef _SECOND_ORDER_PRUNING_
		while (!Qe.empty())
			Qe.pop();
		old_removed_edges_n = removed_edges_n;
#endif

		// choosing 0 as branching vertex
		if (S_end == 0 && SR_rid[root_v] < R_end && !move_u_to_S_with_prune(root_v, S_end, R_end, level))
		{
			// printf("here1\n");
			restore_SR_and_edges(S_end, R_end, old_S_end, old_R_end, level, old_removed_edges_n);
			return;
		}
		if (PARA.UBR2)
		{
			ui S2_n = 0;
			for (ui i = 0; i < S_end; i++)
				if (R_end - degree[SR[i]] > K)
					S2[S2_n++] = SR[i];

			if (S2_n >= 2)
			{
				collect_removable_vertices_based_on_total_edges(S2_n, S_end, R_end, level);
				if (!remove_vertices_and_edges_with_prune(S_end, R_end, level))
				{
					restore_SR_and_edges(S_end, R_end, old_S_end, old_R_end, level, old_removed_edges_n);
					return;
				}
			}
		}

#ifndef NDEBUG
		for (ui i = 0; i < R_end; i++)
		{
			ui d1 = 0, d2 = 0;
			for (ui j = 0; j < S_end; j++)
				if (matrix[SR[i] * n + SR[j]])
					++d1;
			d2 = d1;
			for (ui j = S_end; j < R_end; j++)
				if (matrix[SR[i] * n + SR[j]])
					++d2;
			assert(d1 == degree_in_S[SR[i]]);
			assert(d2 == degree[SR[i]]);
		}
		for (ui i = 0; i < S_end; i++)
			assert(degree_in_S[SR[i]] + K >= S_end);
		for (ui i = S_end; i < R_end; i++)
			assert(degree_in_S[SR[i]] + K > S_end);
		for (ui i = 0; i < S_end; i++)
			if (degree_in_S[SR[i]] + K == S_end)
			{
				char *t_matrix = matrix + SR[i] * n;
				for (ui j = S_end; j < R_end; j++)
					assert(t_matrix[SR[j]]);
			}
		for (ui i = 0; i < R_end; i++)
			assert(level_id[SR[i]] > level);
#endif

		// greedily add vertices to S
		if (!greedily_add_vertices_to_S(S_end, R_end, level))
		{
			// printf("here2\n");
			restore_SR_and_edges(S_end, R_end, old_S_end, old_R_end, level, old_removed_edges_n);
			return;
		}

#ifndef NDEBUG
		for (ui i = 0; i < R_end; i++)
		{
			ui d1 = 0, d2 = 0;
			for (ui j = 0; j < S_end; j++)
				if (matrix[SR[i] * n + SR[j]])
					++d1;
			d2 = d1;
			for (ui j = S_end; j < R_end; j++)
				if (matrix[SR[i] * n + SR[j]])
					++d2;
			assert(d1 == degree_in_S[SR[i]]);
			assert(d2 == degree[SR[i]]);
		}
		for (ui i = 0; i < S_end; i++)
			assert(degree_in_S[SR[i]] + K >= S_end);
		for (ui i = S_end; i < R_end; i++)
			assert(degree_in_S[SR[i]] + K > S_end);
		for (ui i = 0; i < S_end; i++)
			if (degree_in_S[SR[i]] + K == S_end)
			{
				char *t_matrix = matrix + SR[i] * n;
				for (ui j = S_end; j < R_end; j++)
					assert(t_matrix[SR[j]]);
			}
		for (ui i = 0; i < R_end; i++)
			assert(level_id[SR[i]] > level);
#endif

		if (S_end > lb)
			store_solution(S_end);
		if (R_end > lb && is_kplex(R_end))
			store_solution(R_end);
		if (R_end <= lb + 1 || lb >= _UB_)
		{
			// printf("here3\n");
			restore_SR_and_edges(S_end, R_end, old_S_end, old_R_end, level, old_removed_edges_n);
			return;
		}

		ui beta = lb - S_end;
		if (PARA.bounding != "None")
		{

			if (PARA.bounding == "S-Bound" and bound(S_end, R_end) >= R_end)
			{
				restore_SR_and_edges(S_end, R_end, old_S_end, old_R_end, level, old_removed_edges_n);
				return;
			}

			if (PARA.bounding == "SR-Bound" and CSIZE > 3 * beta && seesawUB(S_end, R_end) <= lb)
			{
				restore_SR_and_edges(S_end, R_end, old_S_end, old_R_end, level, old_removed_edges_n);
				return;
			}

			if (PARA.bounding == "R-Bound" and colorUB(S_end, R_end) <= lb)
			{
				restore_SR_and_edges(S_end, R_end, old_S_end, old_R_end, level, old_removed_edges_n);
				return;
			}

			if (PARA.bounding == "AltRB")
			{
				int ub = altRB(S_end, R_end, level);
				if (ub <= lb)
				{
					restore_SR_and_edges(S_end, R_end, old_S_end, old_R_end, level, old_removed_edges_n);
					return;
				}
			}
		}

#ifndef NDEBUG
		for (ui i = 0; i < R_end; i++)
		{
			ui d1 = 0, d2 = 0;
			for (ui j = 0; j < S_end; j++)
				if (matrix[SR[i] * n + SR[j]])
					++d1;
			d2 = d1;
			for (ui j = S_end; j < R_end; j++)
				if (matrix[SR[i] * n + SR[j]])
					++d2;
			assert(d1 == degree_in_S[SR[i]]);
			assert(d2 == degree[SR[i]]);
		}
		for (ui i = 0; i < S_end; i++)
			assert(degree_in_S[SR[i]] + K >= S_end);
		for (ui i = S_end; i < R_end; i++)
			assert(degree_in_S[SR[i]] + K > S_end);
		for (ui i = 0; i < S_end; i++)
			if (degree_in_S[SR[i]] + K == S_end)
			{
				char *t_matrix = matrix + SR[i] * n;
				for (ui j = S_end; j < R_end; j++)
					assert(t_matrix[SR[j]]);
			}
		for (ui i = 0; i < R_end; i++)
			assert(level_id[SR[i]] > level);
#endif

		if (PARA.branching == "Binary-Br")
		{

			// First branch moves u to S
			ui pre_lb = lb, t_old_S_end = S_end, t_old_R_end = R_end, t_old_removed_edges_n = 0;
			ui u = SR[S_end];
			if (move_u_to_S_with_prune(u, S_end, R_end, level))
				BB_search(S_end, R_end, level + 1);

			// the second branch exclude u from G
			// This version of 2nd branch restores u through remove_u function
			{
				restore_SR_and_edges(S_end, R_end, S_end, t_old_R_end, level, t_old_removed_edges_n);
				while (!Qv.empty())
				{
					ui v = Qv.front();
					Qv.pop();
					level_id[v] = n;
				}
				B.clear();

				bool succeed = remove_u_from_S_with_prune(S_end, R_end, level);
				if (succeed && lb > pre_lb)
					succeed = collect_removable_vertices_and_edges(S_end, R_end, level);
				if (remove_vertices_and_edges_with_prune(S_end, R_end, level))
					BB_search(S_end, R_end, level + 1);
			}
			restore_SR_and_edges(S_end, R_end, old_S_end, old_R_end, level, old_removed_edges_n);
			return;
		}

		else if (PARA.branching == "S-Br" or PARA.branching == "R-Br" or PARA.branching == "SR-Br" or (K <= 5 && sparse && PARA.branching == "Adaptive-Br"))
		{

			// ******************* Adding our branching stuff here...
			ui t_R_end = R_end;

			if (PARA.branching == "R-Br")
				R_end = R_branching(S_end, R_end, level);
			else if (PARA.branching == "SR-Br")
				R_end = SR_branching(S_end, R_end, level);
			else // default S-Branching
				R_end = S_branching(S_end, R_end, level);
			while (R_end < t_R_end)
			{
				// branching vertices are now in R_end to t_R_end, and they are already sorted in peelOrder
				// move branching vertex back to C
				ui u = SR[R_end];
				assert(level_id[u] == level && SR_rid[u] == R_end);
				R_end++;
				level_id[u] = n;
				char *t_matrix = matrix + u * n;
				degree[u] = degree_in_S[u] = 0;
				for (ui i = 0; i < R_end; i++)
					if (t_matrix[SR[i]])
					{
						ui w = SR[i];
						++degree[w];
						++degree[u];
						if (i < S_end)
							++degree_in_S[u];
					}

				if (lb >= _UB_)
					return;
				if (level == 1)
					found_larger = false;
				if (found_larger)
					continue;

				ui pre_lb = lb, t_old_S_end = S_end, t_old_R_end = R_end, t_old_removed_edges_n = 0;
#ifdef _SECOND_ORDER_PRUNING_
				if (ctcp_enabled)
				{
					while (!Qe.empty())
						Qe.pop();
					t_old_removed_edges_n = removed_edges_n;
				}
#endif
				if (move_u_to_S_with_prune(u, S_end, R_end, level))
					BB_search(S_end, R_end, level + 1);
				restore_SR_and_edges(S_end, R_end, t_old_S_end, t_old_R_end, level, t_old_removed_edges_n);
			}

			restore_SR_and_edges(S_end, R_end, old_S_end, old_R_end, level, old_removed_edges_n);
		}
		else
		{ // pivot based branching
			if (B.empty() || SR_rid[B.back()] >= R_end || SR_rid[B.back()] < S_end)
				branch(S_end, R_end);
			if (B.empty())
			{
				restore_SR_and_edges(S_end, R_end, old_S_end, old_R_end, level, old_removed_edges_n);
				return;
			}
			ui u = B.back();
			B.pop_back();

			// First branch moves u to S
			ui pre_lb = lb, t_old_S_end = S_end, t_old_R_end = R_end, t_old_removed_edges_n = 0;
#ifdef _SECOND_ORDER_PRUNING_
			if (ctcp_enabled)
			{
				while (!Qe.empty())
					Qe.pop();
				t_old_removed_edges_n = removed_edges_n;
			}
#endif
			if (move_u_to_S_with_prune(u, S_end, R_end, level))
				BB_search(S_end, R_end, level + 1);

			// the second branch exclude u from G
			// This version of 2nd branch restores u through remove_u function
			{
				restore_SR_and_edges(S_end, R_end, S_end, t_old_R_end, level, t_old_removed_edges_n);
				while (!Qv.empty())
				{
					ui v = Qv.front();
					Qv.pop();
					level_id[v] = n;
				}
				B.clear();
#ifdef _SECOND_ORDER_PRUNING_
				while (!Qe.empty())
					Qe.pop();
				t_old_removed_edges_n = removed_edges_n;
#endif
				bool succeed = remove_u_from_S_with_prune(S_end, R_end, level);
				if (succeed && lb > pre_lb)
					succeed = collect_removable_vertices_and_edges(S_end, R_end, level);
				if (remove_vertices_and_edges_with_prune(S_end, R_end, level))
					BB_search(S_end, R_end, level + 1);
			}
			restore_SR_and_edges(S_end, R_end, old_S_end, old_R_end, level, old_removed_edges_n);
		}
	}
	void collect_removable_vertices_based_on_total_edges(ui S2_n, ui S_end, ui R_end, ui level)
	{
		vp.resize(R_end - S_end);
		ui max_nn = 0;
		for (ui i = S_end; i < R_end; i++)
		{
			ui nn = 0;
			if (S2_n != S_end)
			{
				char *t_matrix = matrix + SR[i] * n;
				for (ui j = 0; j < S2_n; j++)
					if (!t_matrix[S2[j]])
						++nn;
			}
			else
				nn = S_end - degree_in_S[SR[i]];
			if (nn > max_nn)
				max_nn = nn;
			vp[i - S_end].first = SR[i];
			vp[i - S_end].second = nn;
		}
		ui *cnt = neighbors;
		for (ui i = 0; i <= max_nn; i++)
			cnt[i] = 0;
		for (ui i = 0; i < vp.size(); i++)
			++cnt[vp[i].second];
		for (ui i = 0; i < max_nn; i++)
			cnt[i + 1] += cnt[i];
		for (ui i = max_nn; i > 0; i--)
			cnt[i] = cnt[i - 1];
		cnt[0] = 0;
		ui *ids = nonneighbors;
		for (ui i = 0; i < vp.size(); i++)
			ids[cnt[vp[i].second]++] = i;

		ui total_support = 0;
		for (ui i = 0; i < S2_n; i++)
			total_support += K - S_end + degree_in_S[S2[i]];

		ui new_n = 0;
		while (!Qv.empty())
			Qv.pop();
		for (ui i = 0; i < vp.size(); i++)
		{
			ui idx = ids[i], v = vp[ids[i]].first;
			ui t_support = total_support - vp[idx].second;
			char *t_matrix = matrix + v * n;
			ui j = 0, v_support = K - 1 - S_end + degree_in_S[v], ub = S_end + 1;
			// ISc.clear();
			while (true)
			{
				if (j == new_n)
					j = i + 1;
				if (j >= vp.size() || ub > lb || ub + vp.size() - j <= lb)
					break;
				ui u = vp[ids[j]].first, nn = vp[ids[j]].second;
				if (t_support < nn)
					break;
				if (t_matrix[u])
				{
					t_support -= nn;
					++ub;
					// ISc.push_back(u);
				}
				else if (v_support > 0)
				{
					--v_support;
					t_support -= nn;
					++ub;
					// ISc.push_back(u);
				}
				++j;
			}
			// if(ub > lb) {
			// 	ub=bound(S_end, R_end, v, ISc);
			// 	if(flag and ub<=lb) cout<<ub<<" "<<lb<<endl;
			// }
			if (ub <= lb)
			{
				level_id[v] = level;
				Qv.push(v);
			}
			else
				ids[new_n++] = ids[i];
		}
	}
	ui support(ui S_end, ui u)
	{
		return K - (S_end - degree_in_S[u]);
	}

	bool greedily_add_vertices_to_S(ui &S_end, ui &R_end, ui level)
	{
		while (true)
		{
			ui *candidates = S2;
			ui candidates_n = 0;
			ui skip;
			if (all_kplex_search)
				skip = 1;
			else
				skip = 2;
			for (ui i = S_end; i < R_end; i++)
			{
				ui u = SR[i];
				if (R_end - degree[u] > K)
					continue;

				if (degree[u] >= R_end - skip)
				{
					candidates[candidates_n++] = u;
					continue;
				}

				char *t_matrix = matrix + u * n;
				bool OK = true;
				for (ui j = 0; j < R_end; j++)
					if (j != i && !t_matrix[SR[j]] && R_end - degree[SR[j]] > K)
					{
						OK = false;
						break;
					}
				if (OK)
					candidates[candidates_n++] = u;
			}

			if (!candidates_n)
				break;

			while (candidates_n)
			{
				ui u = candidates[--candidates_n];
				assert(SR_rid[u] >= S_end);
				if (SR_rid[u] >= R_end)
					return false;

				if (!move_u_to_S_with_prune(u, S_end, R_end, level))
					return false;
			}
		}
		return true;
	}

	void check_degrees(ui S_end, ui R_end)
	{
		for (ui i = 0; i < R_end; i++)
		{
			ui d1 = 0, d2 = 0;
			for (ui j = 0; j < S_end; j++)
				if (matrix[SR[i] * n + SR[j]])
					++d1;
			d2 = d1;
			for (ui j = S_end; j < R_end; j++)
				if (matrix[SR[i] * n + SR[j]])
					++d2;
			assert(d1 == degree_in_S[SR[i]]);
			assert(d2 == degree[SR[i]]);
		}
	}

	bool greedily_add_nonneighbors(ui *candidates, ui candidates_n, ui &S_end, ui &R_end, ui level)
	{
		while (candidates_n)
		{
			ui u = candidates[--candidates_n];
			assert(SR_rid[u] >= S_end);
			if (SR_rid[u] >= R_end || !move_u_to_S_with_prune(u, S_end, R_end, level))
				return false;
		}
		return true;
	}

	void get_neighbors_and_nonneighbors(ui u, ui R_end, ui &neighbors_n, ui &nonneighbors_n)
	{
		neighbors_n = 0;
		nonneighbors_n = 0;
		char *t_matrix = matrix + u * n;
		for (ui i = 0; i < R_end; i++)
			if (SR[i] != u)
			{
				if (t_matrix[SR[i]])
					neighbors[neighbors_n++] = SR[i];
				else
					nonneighbors[nonneighbors_n++] = SR[i];
			}
	}

	bool move_u_to_S_with_prune(ui u, ui &S_end, ui &R_end, ui level)
	{
		assert(SR_rid[u] >= S_end && SR_rid[u] < R_end && SR[SR_rid[u]] == u);
		assert(degree_in_S[u] + K > S_end);
#ifndef NDEBUG
		for (ui i = 0; i < S_end; i++)
			assert(degree_in_S[SR[i]] + K >= S_end);
		for (ui i = S_end; i < R_end; i++)
			assert(degree_in_S[SR[i]] + K > S_end);
		for (ui i = 0; i < R_end; i++)
			assert(level_id[SR[i]] > level);
#endif
		if (SR_rid[u] != S_end)
			swap_pos(S_end, SR_rid[u]);
		++S_end;

		ui neighbors_n = 0, nonneighbors_n = 0;
		get_neighbors_and_nonneighbors(u, R_end, neighbors_n, nonneighbors_n);
		assert(neighbors_n + nonneighbors_n == R_end - 1);
		for (ui i = 0; i < neighbors_n; i++)
			++degree_in_S[neighbors[i]];
#ifndef NDEBUG
		for (ui i = 0; i < S_end; i++)
			assert(degree_in_S[SR[i]] + K >= S_end);
#endif

		while (!Qv.empty())
			Qv.pop();
		// reduction rules based on the fact that each vertex can have at most k-1 nonneighbors
		for (ui i = 0; i < nonneighbors_n; i++)
		{
			ui v = nonneighbors[i];
			if (SR_rid[v] >= S_end)
			{
				if (level_id[v] == level)
					continue;
				if ((S_end - degree_in_S[v] >= K) || (S_end - degree_in_S[u] == K))
				{
					level_id[v] = level;
					Qv.push(v);
				}
			}
			else if (S_end - degree_in_S[v] == K)
			{
				char *tt_matrix = matrix + v * n;
				for (ui j = S_end; j < R_end; j++)
					if (level_id[SR[j]] > level && !tt_matrix[SR[j]])
					{
						level_id[SR[j]] = level;
						Qv.push(SR[j]);
					}
			}
		}

#ifndef NDEBUG
		for (ui i = 0; i < S_end; i++)
			if (degree_in_S[SR[i]] + K == S_end)
			{
				char *t_matrix = matrix + SR[i] * n;
				for (ui j = S_end; j < R_end; j++)
					assert(level_id[SR[j]] == level || t_matrix[SR[j]]);
			}
#endif

#ifdef _SECOND_ORDER_PRUNING_
		for (ui i = 0; i < nonneighbors_n; i++)
		{
			int v = nonneighbors[i];
			if (SR_rid[v] < S_end || level_id[v] == level)
				continue;
			if (upper_bound_based_prune(S_end, u, v))
			{
				level_id[v] = level;
				Qv.push(v);
			}
		}

		// update cn(.,.) for pairs of u's neighbors
		for (ui i = 0; i < neighbors_n; i++)
			for (ui j = i + 1; j < neighbors_n; j++)
			{
				assert(cn[neighbors[i] * n + neighbors[j]]);
				--cn[neighbors[i] * n + neighbors[j]];
				--cn[neighbors[j] * n + neighbors[i]];
			}

		while (!Qe.empty())
			Qe.pop();
		ui new_n = 0;
		for (ui i = 0; i < nonneighbors_n; i++)
			if (level_id[nonneighbors[i]] > level)
				nonneighbors[new_n++] = nonneighbors[i];
		nonneighbors_n = new_n;
		for (ui i = 1; i < nonneighbors_n; i++)
		{ // process pairs of u's non-neighbors
			ui w = nonneighbors[i];
			for (ui j = 0; j < i; j++)
			{
				ui v = nonneighbors[j];
				if (!upper_bound_based_prune(S_end, v, w))
					continue;
				assert(SR_rid[w] > SR_rid[v]);
				if (SR_rid[w] < S_end)
					return false; // v, w \in S --- pruned
				else if (SR_rid[v] >= S_end)
				{ // v, w, \in R --- the edge (v,w) can be removed
					if (matrix[v * n + w])
						Qe.push(std::make_pair(v, w));
				}
				else
				{
					assert(level_id[w] > level);
					level_id[w] = level;
					Qv.push(w);
					break;
				}
			}
		}
#endif

		return remove_vertices_and_edges_with_prune(S_end, R_end, level);
	}

	bool remove_vertices_and_edges_with_prune(ui S_end, ui &R_end, ui level)
	{
#ifdef _SECOND_ORDER_PRUNING_
		while (!Qv.empty() || !Qe.empty())
		{
#else
		while (!Qv.empty())
		{
#endif
			while (!Qv.empty())
			{
				ui u = Qv.front();
				Qv.pop(); // remove u
				assert(SR[SR_rid[u]] == u);
				assert(SR_rid[u] >= S_end && SR_rid[u] < R_end);
				--R_end;
				swap_pos(SR_rid[u], R_end);

				bool terminate = false;
				ui neighbors_n = 0;
				char *t_matrix = matrix + u * n;
				for (ui i = 0; i < R_end; i++)
					if (t_matrix[SR[i]])
					{
						ui w = SR[i];
						neighbors[neighbors_n++] = w;
						--degree[w];
						if (degree[w] + K <= lb)
						{
							if (i < S_end)
								terminate = true; // UB1
							else if (level_id[w] > level)
							{ // RR3
								level_id[w] = level;
								Qv.push(w);
							}
						}
					}

				// UB1
				if (terminate)
				{
					for (ui i = 0; i < neighbors_n; i++)
						++degree[neighbors[i]];
					level_id[u] = n;
					++R_end;
					return false;
				}

#ifdef _SECOND_ORDER_PRUNING_
				for (ui i = 1; i < neighbors_n; i++)
				{
					ui w = neighbors[i];
					for (ui j = 0; j < i; j++)
					{
						ui v = neighbors[j];
						assert(cn[v * n + w]);
#ifndef NDEBUG
						ui common_neighbors = 0;
						for (ui k = S_end; k <= R_end; k++)
							if (matrix[SR[k] * n + v] && matrix[SR[k] * n + w])
								++common_neighbors;
						assert(cn[v * n + w] == common_neighbors);
						assert(cn[w * n + v] == common_neighbors);
#endif
						--cn[v * n + w];
						--cn[w * n + v];
#ifndef NDEBUG
						common_neighbors = 0;
						for (ui k = S_end; k < R_end; k++)
							if (matrix[SR[k] * n + v] && matrix[SR[k] * n + w])
								++common_neighbors;
						assert(cn[v * n + w] == common_neighbors);
						assert(cn[w * n + v] == common_neighbors);
#endif

						if (!upper_bound_based_prune(S_end, v, w))
							continue;

						if (SR_rid[w] < S_end)
							terminate = true; // v, w \in S --- UB2
						else if (SR_rid[v] >= S_end)
						{ // v, w, \in R --- RR5
							if (matrix[v * n + w])
								Qe.push(std::make_pair(v, w));
						}
						else if (level_id[w] > level)
						{ // RR4
							level_id[w] = level;
							Qv.push(w);
						}
					}
				}
				if (terminate)
					return false;
#endif
			}

#ifdef _SECOND_ORDER_PRUNING_
			if (Qe.empty())
				break;

			ui v = Qe.front().first, w = Qe.front().second;
			Qe.pop();
			if (level_id[v] <= level || level_id[w] <= level || !matrix[v * n + w])
				continue;
			assert(SR_rid[v] >= S_end && SR_rid[v] < R_end && SR_rid[w] >= S_end && SR_rid[w] < R_end);

			if (degree[v] + K <= lb + 1)
			{
				level_id[v] = level;
				Qv.push(v);
			}
			if (degree[w] + K <= lb + 1)
			{
				level_id[w] = level;
				Qv.push(w);
			}
			if (!Qv.empty())
				continue;

#ifndef NDEBUG
			// printf("remove edge between %u and %u\n", v, w);
#endif

			assert(matrix[v * n + w]);
			matrix[v * n + w] = matrix[w * n + v] = 0;
			--degree[v];
			--degree[w];

			if (removed_edges.size() == removed_edges_n)
			{
				removed_edges.push_back(std::make_pair(v, w));
				++removed_edges_n;
			}
			else
				removed_edges[removed_edges_n++] = std::make_pair(v, w);

			char *t_matrix = matrix + v * n;
			for (ui i = 0; i < R_end; i++)
				if (t_matrix[SR[i]])
				{
					--cn[w * n + SR[i]];
					--cn[SR[i] * n + w];
					if (!upper_bound_based_prune(S_end, w, SR[i]))
						continue;
					if (i < S_end)
					{
						if (level_id[w] > level)
						{
							level_id[w] = level;
							Qv.push(w);
						}
					}
					else if (matrix[w * n + SR[i]])
						Qe.push(std::make_pair(w, SR[i]));
				}
			t_matrix = matrix + w * n;
			for (ui i = 0; i < R_end; i++)
				if (t_matrix[SR[i]])
				{
					--cn[v * n + SR[i]];
					--cn[SR[i] * n + v];
					if (!upper_bound_based_prune(S_end, v, SR[i]))
						continue;
					if (i < S_end)
					{
						if (level_id[v] > level)
						{
							level_id[v] = level;
							Qv.push(v);
						}
					}
					else if (matrix[v * n + SR[i]])
						Qe.push(std::make_pair(v, SR[i]));
				}
#endif
		}

		return true;
	}

	void restore_SR_and_edges(ui &S_end, ui &R_end, ui old_S_end, ui old_R_end, ui level, ui old_removed_edges_n)
	{
#ifndef NDEBUG
		for (ui i = 0; i < R_end; i++)
		{
			ui d1 = 0, d2 = 0;
			for (ui j = 0; j < S_end; j++)
				if (matrix[SR[i] * n + SR[j]])
					++d1;
			d2 = d1;
			for (ui j = S_end; j < R_end; j++)
				if (matrix[SR[i] * n + SR[j]])
					++d2;
			assert(d1 == degree_in_S[SR[i]]);
			assert(d2 == degree[SR[i]]);
		}
#endif
		while (!Qv.empty())
		{
			ui u = Qv.front();
			Qv.pop();
			assert(level_id[u] == level);
			assert(SR_rid[u] < R_end);
			level_id[u] = n;
		}

#ifndef NDEBUG
		for (ui i = 0; i < R_end; i++)
		{
			if (level_id[SR[i]] <= level)
				printf("level_id[%u] = %u, level = %u, n = %u\n", SR[i], level_id[SR[i]], level, n);
			assert(level_id[SR[i]] > level);
		}
#endif

		for (; R_end < old_R_end; R_end++)
		{ // insert u back into R
			ui u = SR[R_end];
			assert(level_id[u] == level && SR_rid[u] == R_end);
			level_id[u] = n;

			ui neighbors_n = 0;
			char *t_matrix = matrix + u * n;
			degree[u] = degree_in_S[u] = 0;

			for (ui i = 0; i < R_end; i++)
				if (t_matrix[SR[i]])
				{
					ui w = SR[i];
					neighbors[neighbors_n++] = w;
					++degree[w];
					++degree[u];
					if (i < S_end)
						++degree_in_S[u];
				}
#ifdef _SECOND_ORDER_PRUNING_
			for (ui i = 0; i < neighbors_n; i++)
			{
				ui v = neighbors[i];
				for (ui j = i + 1; j < neighbors_n; j++)
				{
					ui w = neighbors[j];
					++cn[v * n + w];
					++cn[w * n + v];
				}
			}
			ui *t_cn = cn + u * n;
			for (ui i = 0; i < R_end; i++)
				t_cn[SR[i]] = 0;
			for (ui i = 0; i < neighbors_n; i++)
				if (SR_rid[neighbors[i]] >= S_end)
				{
					ui v = neighbors[i];
					char *t_matrix = matrix + v * n;
					for (ui j = 0; j < R_end; j++)
						if (t_matrix[SR[j]])
							++t_cn[SR[j]];
				}
			for (ui i = 0; i < R_end; i++)
			{
				cn[SR[i] * n + u] = t_cn[SR[i]];
#ifndef NDEBUG
				ui common_neighbors = 0, v = SR[i], w = u;
				for (ui k = S_end; k < R_end; k++)
					if (matrix[SR[k] * n + v] && matrix[SR[k] * n + w])
						++common_neighbors;
				if (t_cn[SR[i]] != common_neighbors)
					printf("t_cn[SR[i]] = %u, comon_neighbors = %u\n", t_cn[SR[i]], common_neighbors);
				assert(t_cn[SR[i]] == common_neighbors);
#endif
			}
#endif
		}

#ifndef NDEBUG
		for (ui i = 0; i < R_end; i++)
		{
			ui d1 = 0, d2 = 0;
			for (ui j = 0; j < S_end; j++)
				if (matrix[SR[i] * n + SR[j]])
					++d1;
			d2 = d1;
			for (ui j = S_end; j < R_end; j++)
				if (matrix[SR[i] * n + SR[j]])
					++d2;
			assert(d1 == degree_in_S[SR[i]]);
			assert(d2 == degree[SR[i]]);
		}
#endif

		for (; S_end > old_S_end; S_end--)
		{ // move u from S to R
			ui u = SR[S_end - 1];
			assert(SR_rid[u] == S_end - 1);

			ui neighbors_n = 0;
			char *t_matrix = matrix + u * n;
			for (ui i = 0; i < R_end; i++)
				if (t_matrix[SR[i]])
				{
					ui w = SR[i];
					neighbors[neighbors_n++] = w;
					--degree_in_S[w];
				}
#ifdef _SECOND_ORDER_PRUNING_
			for (ui i = 0; i < neighbors_n; i++)
			{
				ui v = neighbors[i];
				for (ui j = i + 1; j < neighbors_n; j++)
				{
					ui w = neighbors[j];
					++cn[v * n + w];
					++cn[w * n + v];
				}
			}
			ui *t_cn = cn + u * n;
			for (ui i = 0; i < R_end; i++)
				t_cn[SR[i]] = 0;
			for (ui i = 0; i < neighbors_n; i++)
				if (SR_rid[neighbors[i]] >= S_end)
				{
					ui v = neighbors[i];
					char *t_matrix = matrix + v * n;
					for (ui j = 0; j < R_end; j++)
						if (t_matrix[SR[j]])
							++t_cn[SR[j]];
				}
			for (ui i = 0; i < R_end; i++)
			{
				cn[SR[i] * n + u] = t_cn[SR[i]];
#ifndef NDEBUG
				ui common_neighbors = 0, v = SR[i], w = u;
				for (ui k = S_end; k < R_end; k++)
					if (matrix[SR[k] * n + v] && matrix[SR[k] * n + w])
						++common_neighbors;
				if (t_cn[SR[i]] != common_neighbors)
					printf("t_cn[SR[i]] = %u, comon_neighbors = %u\n", t_cn[SR[i]], common_neighbors);
				assert(t_cn[SR[i]] == common_neighbors);
#endif
			}
#endif
		}

#ifndef NDEBUG
		for (ui i = 0; i < R_end; i++)
		{
			ui d1 = 0, d2 = 0;
			for (ui j = 0; j < S_end; j++)
				if (matrix[SR[i] * n + SR[j]])
					++d1;
			d2 = d1;
			for (ui j = S_end; j < R_end; j++)
				if (matrix[SR[i] * n + SR[j]])
					++d2;
			assert(d1 == degree_in_S[SR[i]]);
			assert(d2 == degree[SR[i]]);
		}
#endif

#ifdef _SECOND_ORDER_PRUNING_
		for (ui i = old_removed_edges_n; i < removed_edges_n; i++)
		{ // insert edge back into matrix
			ui v = removed_edges[i].first, w = removed_edges[i].second;
			assert(SR_rid[v] >= S_end && SR_rid[v] < R_end && SR_rid[w] >= S_end && SR_rid[w] < R_end);
			if (matrix[v * n + w])
				continue;

#ifndef NDEBUG
			// printf("restore edge between %u and %u\n", v, w);
#endif
			matrix[v * n + w] = matrix[w * n + v] = 1;
			++degree[v];
			++degree[w];

			char *t_matrix = matrix + v * n;
			for (ui i = 0; i < R_end; i++)
				if (t_matrix[SR[i]])
				{
					++cn[w * n + SR[i]];
					++cn[SR[i] * n + w];
				}
			t_matrix = matrix + w * n;
			for (ui i = 0; i < R_end; i++)
				if (t_matrix[SR[i]])
				{
					++cn[v * n + SR[i]];
					++cn[SR[i] * n + v];
				}
		}
		removed_edges_n = old_removed_edges_n;
#endif

#ifndef NDEBUG
		for (ui i = 0; i < R_end; i++)
		{
			ui d1 = 0, d2 = 0;
			for (ui j = 0; j < S_end; j++)
				if (matrix[SR[i] * n + SR[j]])
					++d1;
			d2 = d1;
			for (ui j = S_end; j < R_end; j++)
				if (matrix[SR[i] * n + SR[j]])
					++d2;
			assert(d1 == degree_in_S[SR[i]]);
			assert(d2 == degree[SR[i]]);
		}
		for (ui i = 0; i < S_end; i++)
			assert(degree_in_S[SR[i]] + K >= S_end);
		for (ui i = S_end; i < R_end; i++)
			assert(old_S_end == S_end || degree_in_S[SR[i]] + K > S_end);
		for (ui i = 0; i < S_end; i++)
			if (degree_in_S[SR[i]] + K == S_end)
			{
				char *t_matrix = matrix + SR[i] * n;
				for (ui j = S_end; j < R_end; j++)
					assert(old_S_end == S_end || t_matrix[SR[j]]);
			}
		for (ui i = 0; i < R_end; i++)
			assert(level_id[SR[i]] > level);
#endif
	}

	bool remove_u_from_S_with_prune(ui &S_end, ui &R_end, ui level)
	{
		assert(S_end);
		ui u = SR[S_end - 1];
		--S_end;
		--R_end;
		swap_pos(S_end, R_end);
		level_id[u] = level;

		bool terminate = false;
		ui neighbors_n = 0;
		char *t_matrix = matrix + u * n;
		for (ui i = 0; i < R_end; i++)
			if (t_matrix[SR[i]])
				neighbors[neighbors_n++] = SR[i];
		for (ui i = 0; i < neighbors_n; i++)
		{
			ui v = neighbors[i];
			--degree_in_S[v];
			--degree[v];
			if (degree[v] + K <= lb)
			{
				if (SR_rid[v] < S_end)
					terminate = true;
				else
				{
					assert(level_id[v] > level);
					level_id[v] = level;
					Qv.push(v);
				}
			}
		}
		if (terminate)
			return false;

#ifdef _SECOND_ORDER_PRUNING_
		for (ui i = 1; i < neighbors_n; i++)
			if (level_id[neighbors[i]] > level)
			{
				ui w = neighbors[i];
				for (ui j = 0; j < i; j++)
				{
					ui v = neighbors[j];
					if (!upper_bound_based_prune(S_end, v, w))
						continue;

					assert(SR_rid[v] < SR_rid[w]);
					if (SR_rid[w] < S_end)
						return false; // v, w \in S
					else if (SR_rid[v] >= S_end)
					{ // v, w, \in R
						if (matrix[v * n + w])
							Qe.push(std::make_pair(v, w));
					}
					else
					{
						assert(level_id[w] > level);
						level_id[w] = level;
						Qv.push(w);
						break;
					}
				}
			}
#endif
		return true;
	}

	bool collect_removable_vertices_and_edges(ui S_end, ui R_end, ui level)
	{
		for (ui i = 0; i < S_end; i++)
			if (degree[SR[i]] + K <= lb)
				return false;

#ifdef _SECOND_ORDER_PRUNING_
		for (ui i = 0; i < S_end; i++)
			for (ui j = i + 1; j < S_end; j++)
			{
				if (upper_bound_based_prune(S_end, SR[i], SR[j]))
					return false;
			}
#endif

		for (ui i = S_end; i < R_end; i++)
			if (level_id[SR[i]] > level)
			{
				ui v = SR[i];
				if (S_end - degree_in_S[v] >= K || degree[v] + K <= lb)
				{
					level_id[v] = level;
					Qv.push(v);
					continue;
				}
				char *t_matrix = matrix + v * n;
				for (ui j = 0; j < S_end; j++)
				{
#ifdef _SECOND_ORDER_PRUNING_
					if ((S_end - degree_in_S[SR[j]] == K && !t_matrix[SR[j]]) || upper_bound_based_prune(S_end, v, SR[j]))
#else
					if (S_end - degree_in_S[SR[j]] == K && !t_matrix[SR[j]])
#endif
					{
						level_id[v] = level;
						Qv.push(v);
						break;
					}
				}
			}

#ifdef _SECOND_ORDER_PRUNING_
		for (ui i = S_end; i < R_end; i++)
			if (level_id[SR[i]] > level)
			{
				for (ui j = i + 1; j < R_end; j++)
					if (level_id[SR[j]] > level && matrix[SR[i] * n + SR[j]])
					{
						if (upper_bound_based_prune(S_end, SR[i], SR[j]))
							Qe.push(std::make_pair(SR[i], SR[j]));
					}
			}
#endif

		return true;
	}

#ifdef _SECOND_ORDER_PRUNING_
	bool upper_bound_based_prune(ui S_end, ui u, ui v)
	{
		// ui ub = S_end + 2*K - (S_end - degree_in_S[u]) - (S_end - degree_in_S[v]) + cn[u*n + v];
		ui ub = 2 * K + degree_in_S[u] - S_end + degree_in_S[v] + cn[u * n + v];
		if (SR_rid[u] >= S_end)
		{
			--ub; // S_end ++
			if (matrix[u * n + v])
				++ub; // degree_in_S[v] ++
		}
		if (SR_rid[v] >= S_end)
		{
			--ub;
			if (matrix[v * n + u])
				++ub;
		}
		return ub <= lb;
	}
#endif

	void swap_pos(ui i, ui j)
	{
		std::swap(SR[i], SR[j]);
		SR_rid[SR[i]] = i;
		SR_rid[SR[j]] = j;
	}

	ui choose_branch_vertex(ui S_end, ui R_end)
	{
		ui *D = neighbors;
		ui D_n = 0;
		for (ui i = 0; i < R_end; i++)
			if (R_end - degree[SR[i]] > K)
				D[D_n++] = SR[i];
		assert(D_n != 0);

		ui min_degree_in_S = n;
		for (ui i = 0; i < D_n; i++)
			if (degree_in_S[D[i]] < min_degree_in_S)
				min_degree_in_S = degree_in_S[D[i]];
		ui u = n, min_degree = n;
		for (ui i = 0; i < D_n; i++)
			if (degree_in_S[D[i]] == min_degree_in_S && degree[D[i]] < min_degree)
			{
				min_degree = degree[D[i]];
				u = D[i];
			}
		assert(u != n);

		if (SR_rid[u] < S_end)
		{
			ui max_degree = 0, b = n;
			char *t_matrix = matrix + u * n;
			for (ui i = S_end; i < R_end; i++)
				if (!t_matrix[SR[i]] && degree[SR[i]] > max_degree)
				{
					max_degree = degree[SR[i]];
					b = SR[i];
				}
			return b;
		}
		else if (degree_in_S[u] < S_end || R_end - degree[u] > K + 1)
			return u;
		else
		{
			ui max_degree = 0, w = n;
			for (ui i = S_end; i < R_end; i++)
				if (degree[SR[i]] > max_degree)
				{
					max_degree = degree[SR[i]];
					w = SR[i];
				}
			if (degree[w] + 1 >= R_end)
			{
				printf("!!! WA degree[w]: %u, R_end: %u\n", degree[w], R_end);
			}
			assert(degree[w] + 1 < R_end);
			if (R_end - degree[w] == 2)
				return w;
			char *t_matrix = matrix + w * n;
			for (ui i = S_end; i < R_end; i++)
				if (!t_matrix[SR[i]] && R_end - degree[SR[i]] == K + 1)
					return SR[i];
		}

		printf("!!! WA in choose_branch_vertex\n");
		return n;
	}

	void branch(ui S_end, ui R_end)
	{
		B.clear();
		ui minnei = 0x3f3f3f3f;
		ui pivot; // should it be 0xffffffff?
		char *t_matrix = matrix + 0 * n;
		for (ui i = S_end; i < R_end; i++)
		{
			ui v = SR[i];
			if (!t_matrix[v] && // HOP2 first
				(degree[v] + K <= lb + 1 ||
				 support(S_end, v) == 1 ||
				 support(S_end, 0) == 1))
			{
				B.push_back(v);
				return;
			}
			if (degree[v] < minnei)
			{
				minnei = degree[v];
				pivot = v;
			}
		}

		t_matrix = matrix + pivot * n;
		for (ui i = S_end; i < R_end; i++)
			if (!t_matrix[SR[i]])
				B.push_back(SR[i]);

		auto comp = [&](int a, int b)
		{ return degree[a] > degree[b]; };
		std::sort(B.begin(), B.end(), comp);
	}

	ui SR_branching(ui S_end, ui R_end, ui level)
	{
		ui cend = R_end;
		ui beta = lb - S_end;
		ui ub = 0;
		bool flag = true;
		do
		{
			double ubp = 0;
			if (flag)
			{
				ubp = tryPartition(S_end, cend);
				if (ubp == 0)
					flag = false;
			}
			double ubc = tryColor(S_end, cend);

			if (ubp == 0 or
				(ISc.size() / ubc > PIMax.size() / ubp) or
				((ISc.size() / ubc == PIMax.size() / ubp) and (ISc.size() > PIMax.size())))

			{
				if (ubc <= beta)
					beta -= ubc;
				else
					break;
				for (ui v : ISc)
					swap_pos(v, --cend);
			}
			else
			{
				if (ubp <= beta)
					beta -= ubp;
				else
					break;
				for (ui v : PIMax)
					swap_pos(v, --cend);
			}
		} while (beta > 0 && cend > S_end);
		if (beta > 0)
			cend -= min(beta, cend - S_end);
		vector<ui> B;
		B.reserve(cend - S_end);
		for (ui i = S_end; i < cend; i++)
			B.push_back(SR[i]);
		return move_candidates_to_end(S_end, R_end, B, level);
		// return move_candidates_to_end(S_end, cend, R_end, level);
	}
	ui R_branching(ui S_end, ui R_end, ui level)
	{

		ui cend = R_end;
		ui beta = lb - S_end;
		vector<ui> B, nextB;
		B.reserve(R_end - S_end);
		nextB.reserve(R_end - S_end);
		for (ui i = S_end; i < R_end; i++)
			B.push_back(SR[i]);
		while (beta > 0 && !B.empty())
		{
			ui ub = tryColor(S_end, B);
			if (ub <= beta)
			{
				bmp.setup(ISc, n);
				nextB.clear();
				for (ui v : B)
					if (!bmp.test(v))
						nextB.push_back(v);
				B = nextB;
				beta -= ub;
			}
			else
				break;
		}
		if (beta > 0)
		{
			ui dec = std::min(beta, (ui)B.size());
			B.resize(B.size() - dec);
		}

		return move_candidates_to_end(S_end, R_end, B, level);
	}

	ui S_branching(ui S_end, ui R_end, ui level)
	{
		for (ui i = 0; i < S_end; i++)
		{
			ui u = SR[i];
			psz[i] = 0;
			if (support(S_end, u) == 0)
				continue;
			// skipping it, because this is a boundary vertex, and it can't have any non-neighbor candidate
			ui *t_LPI = LPI + i * n;
			char *t_matrix = matrix + u * n;
			for (ui j = S_end; j < R_end; j++)
			{
				ui v = SR[j];
				if (!t_matrix[v])
					// PI[u].push_back(v);
					t_LPI[psz[i]++] = v;
			}
		}
		ui beta = lb - S_end;
		ui cend = R_end;

		while (S_end < cend)
		{
			ui maxpi = -1;
			double maxdise = 0;
			for (ui i = 0; i < S_end; i++)
			{
				ui u = SR[i];
				if (psz[i] == 0)
					continue;
				double cost = min(support(S_end, u), psz[i]);
				double dise = psz[i] / cost;
				if (cost <= beta and dise > maxdise)
					maxpi = i, maxdise = dise;
			}
			if (maxpi == -1)
				break;
			else
			{
				// remove pi* from C
				ui *t_LPI = LPI + maxpi * n;
				for (ui i = 0; i < psz[maxpi]; i++)
				{
					// removing from C
					ui v = t_LPI[i];
					swap_pos(SR_rid[v], --cend);
				}
				// beta-=cost(pi*)
				beta -= min(support(S_end, SR[maxpi]), psz[maxpi]);

				// remove maxpi from every pi
				psz[maxpi] = 0;
				for (ui i = 0; i < S_end; i++)
				{
					ui j = 0;
					ui *t_LPI = LPI + i * n;
					for (ui k = 0; k < psz[i]; k++)
						if (SR_rid[t_LPI[k]] < cend)
							t_LPI[j++] = t_LPI[k];
					psz[i] = j;
				}
			}
			if (beta == 0)
				break;
		}
		if (beta > 0)
			cend -= min(beta, cend - S_end);
		vector<ui> B;
		B.reserve(cend - S_end);
		for (ui i = S_end; i < cend; i++)
			B.push_back(SR[i]);
		return move_candidates_to_end(S_end, R_end, B, level);
		// return move_candidates_to_end(S_end, cend, R_end, level);
	}

	ui move_candidates_to_end(ui S_end, ui R_end, vector<ui> &B, ui level)
	{
		std::sort(B.begin(), B.end(), [&](ui u, ui v)
				  { return peelOrder[u] < peelOrder[v]; });

		for (ui u : B)
		{
			swap_pos(SR_rid[u], --R_end);
			level_id[u] = level;
			char *t_matrix = matrix + u * n;
			degree[u] = degree_in_S[u] = 0;
			for (ui i = 0; i < R_end; i++)
			{
				ui w = SR[i];
				// if(level_id[w]==level) continue;
				if (t_matrix[w])
					--degree[w];
			}
		}
		return R_end;
	}
	ui TISUB(ui S_end)
	{
		std::vector<ui> tmp = ISc;

		std::sort(tmp.begin(), tmp.end(), [&](ui a, ui b)
				  { return support(S_end, a) > support(S_end, b); });

		ui ub = 0;

		for (ui i = 0; i < tmp.size(); i++)
		{
			if (support(S_end, tmp[i]) >= i + 1)
				ub = i + 1;
			else
				break;
		}

		return ub;
	}

	ui tryColor(ui S_end, vector<ui> &B)
	{
		ISc.clear();
		for (ui u : B)
		{
			bool flag = true;
			for (ui v : ISc)
				if (matrix[u * n + v])
				{
					flag = false;
					break;
				}
			if (flag)
				ISc.push_back(u);
		}

		return TISUB(S_end);
	}
	ui colorUB(ui S_end, ui R_end)
	{
		ui UB = S_end;
		while (R_end > S_end && UB <= lb)
		{
			double ubc = tryColor(S_end, R_end);
			for (ui v : ISc)
				swap_pos(v, --R_end);
			UB += ubc;
		}
		return UB;
	}

	ui seesawUB(ui S_end, ui R_end)
	{
		ui UB = S_end;
		while (R_end > S_end)
		{
			double ubp = tryPartition(S_end, R_end);
			double ubc = tryColor(S_end, R_end);
			if (ubp == 0 or
				(ISc.size() / ubc > PIMax.size() / ubp) or
				((ISc.size() / ubc == PIMax.size() / ubp) and (ISc.size() > PIMax.size())))

			{
				for (ui v : ISc)
					swap_pos(v, --R_end);
				UB += ubc;
			}
			else
			{

				for (ui v : PIMax)
					swap_pos(v, --R_end);
				UB += ubp;
			}
		}
		return UB;
	}

	void createIS(ui S_end, ui R_end)
	{
		ISc.clear();
		for (ui i = S_end; i < R_end; i++)
		{
			bool flag = true;
			for (ui j : ISc)
				if (is_neigh(i, j))
				{
					flag = false;
					break;
				}
			if (flag)
				ISc.push_back(i);
		}
	}

	ui tryColor(ui S_end, ui R_end)
	{
		createIS(S_end, R_end);
		ui ub = TISUB(S_end);
		return ub;
		ui vlc = 0;
		// collect loose vertices i.e. v \in ISc | support(v) > ub
		for (ui i = 0; i < ISc.size(); i++)
		{
			if (support(S_end, SR[ISc[i]]) > ub)
			{
				std::swap(ISc[i], ISc[vlc]);
				vlc++;
			}
		}
		// ISc[0... vlc) we have loose vertices

		// Lookup inIS(&lookup, &ISc, true);
		bmp.setup(ISc, n);
		for (ui i = S_end; vlc < ub and i < R_end; i++)
		{
			if (bmp.test(i)) // this loop running for C\ISc
				continue;
			ui vc = 0;
			for (ui j = vlc; j < ISc.size(); j++) // this loop runs in ISc\LC
			{
				if (is_neigh(i, ISc[j]))
				{
					std::swap(ISc[vlc + vc], ISc[j]);
					vc++;
				}
			}
			if (vlc + vc + 1 <= ub)
			{
				vlc += vc;
				ISc.push_back(i);
				std::swap(ISc.back(), ISc[vlc++]);
				bmp.set(i);
			}
		}

		for (ui i = S_end; i < R_end; i++)
		{
			if (bmp.test(i) or support(S_end, SR[i]) >= ub) // this loop running for C\ISc
				continue;
			ui nv = 0;
			for (ui j : ISc)
			{
				if (is_neigh(i, j))
					nv++;
			}

			if (nv + support(S_end, SR[i]) <= ub)
			{
				ISc.push_back(i);
				bmp.set(i);
			}
		}
		return ub;
	}
	bool is_neigh(ui i, ui j)
	{
		return matrix[SR[i] * n + SR[j]];
	}
	bool is_neigh_uv(ui u, ui v)
	{
		return matrix[u * n + v];
	}
	ui tryPartition(ui S_end, ui R_end)
	{
		double maxdise = 0;
		ui ub = 0;
		PIMax.clear();
		for (ui i = 0; i < S_end; i++)
		{
			if (support(S_end, SR[i]) == 0)
				continue;
			PI.clear();
			for (ui j = S_end; j < R_end; j++)
			{
				if (!is_neigh(i, j))
					PI.push_back(j);
			}
			if (PI.empty())
				continue;
			ui cost = min(support(S_end, SR[i]), (ui)PI.size());
			double dise = (double)PI.size() / (double)cost;
			if (dise > maxdise or (dise == maxdise and PI.size() > PIMax.size()))
			{
				maxdise = dise, ub = cost;
				std::swap(PI, PIMax);
			}
		}
		return ub;
	}
	ui bound(ui S_end, ui R_end)
	{
		vp.clear();
		for (ui i = 0; i < S_end; i++)
			vp.push_back(std::make_pair(support(S_end, SR[i]), SR[i]));
		// for(ui i = 0;i < S_end;i ++) vp.push_back(std::make_pair(-(degree_in_S[SR[i]]-neiInP[SR[i]]), SR[i]));
		sort(vp.begin(), vp.end());
		ui UB = S_end, cursor = S_end;
		for (ui i = 0; i < (ui)vp.size(); i++)
		{
			ui u = vp[i].second;
			if (vp[i].first == 0)
				continue; // boundary vertex
			ui nn = 0;
			char *t_matrix = matrix + u * n;
			for (ui j = cursor; j < R_end; j++)
				if (!t_matrix[SR[j]])
				{
					if (j != cursor + nn)
						swap_pos(j, cursor + nn);
					++nn;
				}
			ui t_ub = min(nn, vp[i].first);
			// ui t_ub = nn;
			// if(vp[i].first < t_ub) t_ub = vp[i].first;
			if (UB + t_ub <= lb)
			{
				UB += t_ub;
				cursor += nn;
			}
			else
			{
				return cursor + (lb - UB);
			}
		}
		cursor += (lb - UB);
		if (cursor > R_end)
			cursor = R_end;
		return cursor;
	}

	// ===== AltRB helpers =====

	inline ui count_neighbors_in_vec(ui u, const std::vector<ui> &verts) const
	{
		const char *row = matrix + (long long)u * n;
		ui cnt = 0;
		for (ui v : verts)
			if (row[v])
				++cnt;
		return cnt;
	}

	inline ui count_nonneighbors_in_vec(ui u, const std::vector<ui> &verts) const
	{
		return (ui)verts.size() - count_neighbors_in_vec(u, verts);
	}

	// Remove a vertex from the current candidate part R (global in-place update).
	// Safe to call only when v is currently in [S_end, R_end).
	inline void remove_candidate_from_R(ui v, ui &S_end, ui &R_end)
	{
		ui pos = SR_rid[v];
		if (pos < S_end || pos >= R_end)
			return;

		--R_end;
		if (pos != R_end)
			swap_pos(pos, R_end);

		// Update degrees of the remaining candidates.
		char *row = matrix + (long long)v * n;
		for (ui i = S_end; i < R_end; ++i)
		{
			ui u = SR[i];
			if (row[u])
				--degree[u];
		}

		degree[v] = 0;
	}

	/*
	 * AltRB port for your array-based branch state.
	 * It mutates (S_end, R_end) in-place and returns the current upper bound.
	 *
	 * Call pattern:
	 *   ui ub = altRB(S_end, R_end, level);
	 *   if (ub <= lb) { restore...; return; }
	 */
	int altRB_(ui &S_end, ui &R_end, ui level)
	{
		altrb_cnt++;

		// ----------------------------
		// LOCAL FIX: rebuild cache
		// ----------------------------
		auto rebuild_cache = [&]()
		{
			for (ui i = 0; i < R_end; ++i)
			{
				ui v = SR[i];
				ui dinS = 0, dinR = 0;
				char *row = matrix + (long long)v * n;

				for (ui j = 0; j < S_end; ++j)
					if (row[SR[j]])
						++dinS;

				for (ui j = S_end; j < R_end; ++j)
					if (row[SR[j]])
						++dinR;

				degree_in_S[v] = dinS;
				degree[v] = dinS + dinR;
			}
		};

		// wrap all returns to keep state consistent
		auto finish = [&](int ans) -> int
		{
			rebuild_cache();
			return ans;
		};

		while (true)
		{
			// ensure clean state before running AltRB logic
			rebuild_cache();

			// Snapshot
			std::vector<ui> activeS(SR, SR + S_end);
			std::vector<ui> Pi0(SR + S_end, SR + R_end);
			std::vector<ui> usefulS;
			usefulS.reserve(S_end);

			long long ub = (long long)S_end;

			// ----------------------------
			// Partition / ComputeUB
			// ----------------------------
			while (!activeS.empty())
			{
				int sel_idx = -1;
				ui sel_sz = 0;
				int sel_allow = 0;

				for (ui i = 0; i < activeS.size();)
				{
					ui v = activeS[i];
					int allow_v = (int)K - (int)(S_end - degree_in_S[v]);

					if (allow_v <= 0)
					{
						for (ui i = S_end; i < R_end; i++)
							if (!is_neigh_uv(SR[i], v))
								cout << "Error";
						activeS[i] = activeS.back();
						activeS.pop_back();
						continue;
					}

					ui sz = count_nonneighbors_in_vec(v, Pi0);

					if (sz <= (ui)allow_v)
					{
						activeS[i] = activeS.back();
						activeS.pop_back();
						continue;
					}

					if (sel_idx == -1 ||
						(unsigned long long)sz * (unsigned long long)sel_allow >
							(unsigned long long)sel_sz * (unsigned long long)allow_v)
					{
						sel_idx = (int)i;
						sel_sz = sz;
						sel_allow = allow_v;
					}

					++i;
				}

				if (sel_idx == -1)
					break;

				ui sel = activeS[sel_idx];
				activeS[sel_idx] = activeS.back();
				activeS.pop_back();

				usefulS.push_back(sel);
				ub += (long long)sel_allow;

				std::vector<ui> nextPi0;
				nextPi0.reserve(Pi0.size());
				char *row = matrix + (long long)sel * n;

				for (ui v : Pi0)
					if (row[v])
						nextPi0.push_back(v);

				Pi0.swap(nextPi0);
			}

			long long ret = ub + (long long)Pi0.size();
			if (ret <= (long long)lb)
				return finish((int)ret);

			// ----------------------------
			// Step 2: Pi0 reduction
			// ----------------------------
			bool changed = false;

			std::vector<char> alive(n, 0);
			std::vector<int> pi0_deg(n, 0);
			std::queue<ui> q;

			for (ui v : Pi0)
				alive[v] = 1;

			const long long LB_Pi0 = (long long)lb + 1 - ub;

			if (LB_Pi0 > (long long)K - 1)
			{
				for (ui v : Pi0)
				{
					ui cnt = 0;
					char *row = matrix + (long long)v * n;
					for (ui w : Pi0)
						if (alive[w] && row[w])
							++cnt;
					pi0_deg[v] = (int)cnt;
				}

				for (ui v : Pi0)
				{
					int threshold = (int)LB_Pi0 + (int)(S_end - degree_in_S[v]) - (int)K;
					if (pi0_deg[v] < threshold)
						q.push(v);
				}

				while (!q.empty())
				{
					ui v = q.front();
					q.pop();
					if (!alive[v])
						continue;

					alive[v] = 0;
					changed = true;

					remove_candidate_from_R(v, S_end, R_end);

					char *row = matrix + (long long)v * n;
					for (ui w : Pi0)
					{
						if (!alive[w] || !row[w])
							continue;

						--pi0_deg[w];
						int threshold = (int)LB_Pi0 + (int)(S_end - degree_in_S[w]) - (int)K;
						if (pi0_deg[w] < threshold)
							q.push(w);
					}
				}

				ui live_pi0 = 0;
				for (ui v : Pi0)
					if (alive[v])
						++live_pi0;

				ret = ub + (long long)live_pi0;
				if (ret <= (long long)lb)
					return finish((int)ret);
			}

			// ----------------------------
			// RR1 + RR2
			// ----------------------------
			std::vector<ui> snapshotC(SR + S_end, SR + R_end);

			ui live_pi0 = 0;
			for (ui v : Pi0)
				if (alive[v])
					++live_pi0;

			for (ui u : snapshotC)
			{
				if (SR_rid[u] < S_end || SR_rid[u] >= R_end)
					continue;

				ui neighbor_cnt = 0, non_neighbor_cnt = 0;

				for (ui v : Pi0)
				{
					if (!alive[v])
						continue;

					if (matrix[(long long)u * n + v])
						++neighbor_cnt;
					else
						++non_neighbor_cnt;
				}

				int loss_u = (int)S_end - (int)degree_in_S[u];

				long long ub_u;

				if (alive[u])
				{
					ub_u = (long long)S_end + 1 + neighbor_cnt +
						   std::min((int)non_neighbor_cnt - 1, (int)K - 1 - loss_u) +
						   (ub - (long long)S_end);

					if (ub_u <= (long long)lb)
					{
						remove_candidate_from_R(u, S_end, R_end);
						alive[u] = 0;
						changed = true;
						continue;
					}

					int nn_useful = 0;
					for (ui x : usefulS)
						if (!matrix[(long long)u * n + x])
							++nn_useful;

					ub_u -= nn_useful;

					if (ub_u <= (long long)lb)
					{
						remove_candidate_from_R(u, S_end, R_end);
						alive[u] = 0;
						changed = true;
					}
				}
				else
				{
					ub_u = (long long)S_end + 1 + neighbor_cnt +
						   std::min((int)non_neighbor_cnt, (int)K - 1 - loss_u) +
						   (ub - (long long)S_end) - 1;

					if (ub_u <= (long long)lb)
					{
						remove_candidate_from_R(u, S_end, R_end);
						changed = true;
						continue;
					}

					int nn_useful = 0;
					for (ui x : usefulS)
						if (!matrix[(long long)u * n + x])
							++nn_useful;

					ub_u = ub_u + 1 - nn_useful;

					if (ub_u <= (long long)lb)
					{
						remove_candidate_from_R(u, S_end, R_end);
						changed = true;
					}
				}
			}

			live_pi0 = 0;
			for (ui v : Pi0)
				if (alive[v])
					++live_pi0;

			ret = ub + (long long)live_pi0;

			if (ret <= (long long)lb)
				return finish((int)ret);

			// RR2
			if (live_pi0 > 0 && ret == (long long)lb + 1)
			{
				std::vector<ui> force;
				for (ui v : Pi0)
					if (alive[v] && SR_rid[v] >= S_end && SR_rid[v] < R_end)
						force.push_back(v);

				for (ui v : force)
				{
					// check if v can be moved to S.
					if (support(S_end, v) == 0)
						return finish(lb);
					for (ui i = 0; i < S_end; i++)
					{
						ui w = SR[i];
						if (support(S_end, w) == 0 && !is_neigh_uv(v, w))
							return finish(lb);
					}
					if (!move_u_to_S_with_prune(v, S_end, R_end, level))
						return finish(lb);
					changed = true;
				}

				if (S_end > lb)
					store_solution(S_end);
				if (R_end > lb && is_kplex(R_end))
					store_solution(R_end);
			}

			if (!changed)
				return finish((int)ret);

			if (R_end <= lb + 1 || lb >= _UB_)
				return finish(lb);
		}
	}

	ui neigh_count(ui u, Set &P)
	{
		char *row = matrix + u * n;
		ui count = 0;
		for (ui v : P)
		{
			if (row[v])
				count++;
		}
		return count;
	}
	ui non_neigh_count(ui u, Set &P)
	{
		char *row = matrix + u * n;
		ui count = 0;
		for (ui v : P)
		{
			if (!row[v])
				count++;
		}
		return count;
	}

	void neigh_intersect(ui u, Set &P)
	{
		char *row = matrix + u * n;
		ui count = 0;
		for (ui v : P)
		{
			if (!row[v])
				P.reset(v);
		}
	}

	void non_neigh_intersect(ui u, Set &P)
	{
		char *row = matrix + u * n;
		ui count = 0;
		for (ui v : P)
		{
			if (row[v])
				P.reset(v);
		}
	}

	void reduce_Pi_0(Set &P, int cnt, Set &C)
	{
		// if |P|<cnt, then P must be reduce to empty; if cnt<=k, then we can't reduce any vertex
		if (cnt <= paramK || P.size() < cnt)
			return;
		double start_core_reduce = get_system_time_microsecond();
		auto &deg = PI; // we reuse the array to decrease time cost
		for (int u : P)
		{
			// int d = A[u].intersect(P);
			int d = neigh_count(u, P);

			if (d < cnt + loss_cnt[u] - paramK)
			{
				for (int v : P)
				{
					if (v == u)
						break;
					if (is_neigh_uv(u, v))
						deg[v]--;
				}
				P.reset(u);
				C.reset(u);
			}
			else
				deg[u] = d;
		}
		int hh = 0, tt = -1;
		ui* q = S2; // faster than std::queue
		for (int u : P)
			if (deg[u] < cnt + loss_cnt[u] - paramK)
			{
				q[++tt] = u;
			}
		while (hh <= tt)
		{
			int u = q[hh++];
			P.reset(u);
			C.reset(u);
			for (int v : P)
			{
				if (is_neigh_uv(u, v))
				{
					--deg[v];
					if (deg[v] + 1 == cnt + loss_cnt[v] - paramK)
					{
						q[++tt] = v;
					}
				}
			}
		}
	}

	int bound_and_reduce(Set &S, Set &C)
	{
		Timer part_timer;
		int initial_C_size = C.size();
		auto copy_S = S;
		auto copy_C = C;
		int S_sz = S.size();
		int ub = S_sz;
		Set useful_S(S.range); // for v in useful_S, Pi_v is generated
		// DisePUB: Partition
		while (copy_S.size())
		{
			int sel = -1, size = 0, allow = 0;
			for (int v : copy_S)
			{
				// int sz = non_A[v].intersect(copy_C);
				int sz = non_neigh_count(v, copy_C);
				int allow_v = (paramK - loss_cnt[v]);
				if (sz <= allow_v) // Pi_i is useless
				{
					copy_S.reset(v);
				}
				else
				{
					if (sel == -1 || size * allow_v < sz * allow) // (size/ub_cnt)<(sz/cnt)
					{
						sel = v;
						size = sz;
						allow = allow_v;
					}
				}
			}
			if (sel == -1)
				break;
			if (lb + 1 > ub && size * 1.0 / allow <= copy_C.size() * 1.0 / (lb + 1 - ub))
				break;
			copy_S.reset(sel);
			useful_S.set(sel);
			ub += allow;
			// copy_C &= A[sel]; // remove the non-neighbors of sel
			neigh_intersect(sel, copy_C);
		}

		// now copy_C = Pi_0
		auto &Pi_0 = copy_C;
#ifndef SeqRB
		reduce_Pi_0(Pi_0, lb + 1 - ub, C);
#endif
		int ret = ub + Pi_0.size();
		if (ret <= lb) // pruned
			return ret;

#ifndef SeqRB
		// RR1
		// assume we need to include at least $h$ vertices from S+Pi_0; $h$=lb+1-(ub-|S|);
		// then for u∈Pi_i, u must has at least $h-k+1$ neighbors from S+Pi_0
		if (paramK > 5)
		{
			Timer t;
			int LB_Pi_0 = lb + 1 - ub;
			if (LB_Pi_0 > paramK - 1)
			{
				auto temp = C;
				temp ^= Pi_0; // temp = C - Pi_0 = Pi_i
				// u∈Pi_i, then u has at least (lb+1-sigma_(min(|Pi_i|, k-|S\N(v_i)|))-(k-1) = neighbor_cnt) neighbors in Pi_0∪S
				for (int u : temp)
				{
					int threshold = LB_Pi_0 - (paramK - 1 - loss_cnt[u]);
					// if (A[u].intersect(Pi_0) < threshold)
					if (neigh_count(u, Pi_0) < threshold)
						C.reset(u);
				}
			}
		}

		{
			Timer t;
			int Pi_0_size = Pi_0.size();
			int ub_Pi_I = ub - S_sz;
			for (int u : C)
			{
				// int neighbor_cnt = Pi_0.intersect(A[u]);
				int neighbor_cnt = neigh_count(u, Pi_0);
				int non_neighbor_cnt = Pi_0_size - neighbor_cnt;
				if (Pi_0[u])
				{
					int ub_u = S_sz + 1 + neighbor_cnt + min(non_neighbor_cnt - 1, paramK - 1 - loss_cnt[u]) + ub_Pi_I;
					if (ub_u <= lb)
					{
						C.reset(u);
						Pi_0.reset(u);
						Pi_0_size--;
						ret--;
						if (ret <= lb)
							return ret;
					}
					else
					{
						// ub_u -= useful_S.intersect(non_A[u]);
						ub_u -= non_neigh_count(u, useful_S);
						if (ub_u <= lb)
						{
							C.reset(u);
							Pi_0.reset(u);
							Pi_0_size--;
							ret--;
						}
					}
				}
				else
				{
					int ub_u = S_sz + 1 + neighbor_cnt + min(non_neighbor_cnt, paramK - 1 - loss_cnt[u]) + ub_Pi_I - 1;
					if (ub_u <= lb)
					{
						C.reset(u);
					}
					else
					{
						// ub_u = ub_u + 1 - useful_S.intersect(non_A[u]);
						ub_u = ub_u + 1 - non_neigh_count(u, useful_S);
						if (ub_u <= lb)
							C.reset(u);
					}
				}
			}
		}
		// RR2
		if (ret == lb + 1 && Pi_0.size())
		{
			S |= Pi_0;
			C ^= Pi_0;
			Timer start_fast_reduce;
			bool S_is_plex, g_is_plex;
			fast_reduction(S, C, g_is_plex, S_is_plex);

			if (!S_is_plex)
				return lb;
			if (g_is_plex)
			{
				update_lb(S, C);
				return lb;
			}
		}
#endif
		if (C.size() < initial_C_size) // goto next iteration
			return bound_and_reduce(S, C);
		return ret;
	}

	void fast_reduction(Set &S, Set &C, bool &g_is_plex, bool &S_is_plex)
	{

		compute_loss_cnt(S, C, S_is_plex);
		if (!S_is_plex)
			return;

		S_is_plex = true;
		// compute degree of subgraph S∪C
		auto V = C;
		V |= S;
		g_is_plex = 1;
		int sz = V.size();
		ui* deg=S2;
		// u is k-satisfied <==> deg[u]+k >= n
		// u is C_near-satisfied <==> deg[u]+k+1 >= n and u in C
		Set satisfied(S.range);
		Set C_near_satisfied(S.range);
		int S_size = S.size();
		for (int v : V)
		{
			// int neighbor_in_C = A[v].intersect(C);
			int neighbor_in_C = neigh_count(v, C);
			// weak reduce: if d[v] + k <= lb, then remove v
			// note that if we change $sz$, it may affect the later process when finding vertices must include;
			// to avoid bugs, we choose not to decrease $sz$
			if (S[v])
			{
				if (S_size + neighbor_in_C + paramK - loss_cnt[v] <= lb)
				{
					S_is_plex = false;
					return;
				}
			}
			else
			{
				// v is in C
				if (S_size + neighbor_in_C + paramK - loss_cnt[v] <= lb)
				{
					V.reset(v);
					C.reset(v);
					continue;
				}
			}
			deg[v] = S_size - loss_cnt[v] + neighbor_in_C;
			if (deg[v] + paramK >= sz)
			{
				satisfied.set(v);
				if (C[v])
					C_near_satisfied.set(v);
			}
			else
			{
				g_is_plex = false;
				if (deg[v] + paramK + 1 == sz && C[v])
					C_near_satisfied.set(v);
			}
		}
		if (g_is_plex)
			return;
		// now we consider the vertices that must be included
		int satis_cnt = satisfied.size();
		bool S_changed = false;
		for (int u : C_near_satisfied)
		{
			bool must_include = 0;
			// case 1
			if (deg[u] + 2 >= sz)
			{
				must_include = 1;
			}
			else
			{
				// we consider the non-neighbors of u (excluding u itself)
				// int satisfied_non_neighbor = satisfied.intersect(non_A[u]);
				// int tot_non_neighbor = non_A[u].intersect(V);
				int satisfied_non_neighbor = non_neigh_count(u, satisfied);
				int tot_non_neighbor = non_neigh_count(u, V);
				if (satisfied[u])
				{
					satisfied_non_neighbor--;
				}
				tot_non_neighbor--;
				// case 2: u is satisfied and all non-neighbors of u are satisfied
				if (satisfied_non_neighbor == tot_non_neighbor)
				{
					if (satisfied[u])
						must_include = 1;
				}
				if (loss_cnt[u] > 0)
					continue;
				// case 3: deg[u]>=n-k-1 and only one non-neighbor of u is un-satisfied
				else if (satisfied_non_neighbor + 1 == tot_non_neighbor)
				{
					must_include = 1;
				}
				// case 4: deg[u]>=n-k-1 and all the un-satisfied non-neighbors form an independent vertex set
				else
				{
					auto non_neighbor = V;
					// non_neighbor &= non_A[u];
					non_neigh_intersect(u, non_neighbor);
					auto &un_satisfied_non_neighbor = non_neighbor;
					un_satisfied_non_neighbor.sub(satisfied);
					// assert(un_satisfied_non_neighbor.size() >= 2);
					bool is_independent = true;
					for (int a : un_satisfied_non_neighbor)
					{
						for (int b : un_satisfied_non_neighbor)
						{
							if (b >= a)
								break;
							if (is_neigh_uv(a, b))
							{
								is_independent = false;
								break;
							}
						}
						if (!is_independent)
							break;
					}
					if (is_independent)
						must_include = 1;
				}
			}
			if (must_include)
			{
				S.set(u);
				C.reset(u);
				S_changed = true;
				break; // each time we only include one vertex that must be included
			}
		}
		// S is changed, so we need to re-compute loss_cnt[] and deg[]
		if (S_changed)
		{
			fast_reduction(S, C, g_is_plex, S_is_plex);
		}
	}

	void update_lb(Set &S)
	{
		int sz = S.size();
		if (sz > lb)
		{
			lb = sz;
			solution.clear();
			for (int v : S)
				solution.insert(v);
			solution = ptr_g->get_ori_vertices(solution);
			assert(solution.size() == lb);
			printf("Find a larger plex : %d\n", sz);
			fflush(stdout);
		}
	}
	/**
	 * @brief plex=S+C
	 */
	void update_lb(Set &S, Set &C)
	{
		int sz = S.size() + C.size();
		if (sz > lb)
		{
			lb = sz;
			solution.clear();
			for (int v : S)
				solution.insert(v);
			for (int v : C)
				solution.insert(v);
			solution = ptr_g->get_ori_vertices(solution);
			assert(solution.size() == lb);
			printf("Find a larger plex : %d\n", sz);
			fflush(stdout);
		}
	}
	/**
	 * @brief compute loss_cnt[] & remove u if S+u is not a plex
	 * @param S_is_plex serve as return
	 */
	void compute_loss_cnt(Set &S, Set &C, bool &S_is_plex)
	{
		for (int v : S)
		{
			// loss_cnt[v] = non_A[v].intersect(S); // v∈S, delta[v] = the number of non-neighbors of v in S
			loss_cnt[v] = non_neigh_count(v, S);
			if (loss_cnt[v] > paramK)
			{
				S_is_plex = false;
				return;
			}
			else if (loss_cnt[v] == paramK) // v has k non-neighbors in S, so all non-neighbors of v can be removed
											// C &= A[v];
				neigh_intersect(v, C);
		}
		S_is_plex = true;
		update_lb(S);
		for (int u : C)
		{
			// loss_cnt[u] = non_A[u].intersect(S); // u∈C, delta[u] = the number of non-neighbors of u in S
			loss_cnt[u] = non_neigh_count(u, S);
			if (loss_cnt[u] >= paramK) // u has at least k non-neighbors in S, so u can be removed
				C.reset(u);
		}
	}

	int altRB(ui &S_end, ui &R_end, ui level)
	{
		Set S(n), C(n);
		for (ui i = 0; i < S_end; i++)
			S.set(SR[i]);
		for (ui i = S_end; i < R_end; i++)
			C.set(SR[i]);
		bool S_is_plex, g_is_plex;
		fast_reduction(S, C, g_is_plex, S_is_plex); // existing techniques

		if (!S_is_plex)
			return lb;
		if (g_is_plex)
		{
			update_lb(S, C);
			return lb;
		}

		int ub = bound_and_reduce(S, C); // AltRB
		if (ub <= lb)
			return ub;

		for (ui i = S_end; i < R_end;)
		{
			ui u = SR[i];
			if (!(C.test(SR[i]) || S.test(SR[i])))
			{
				swap_pos(i, --R_end);
				degree[u] = degree_in_S[u] = 0;
				level_id[u] = level;
			}
			else
			{
				i++;
			}
		}

		S_end = 0;
		for (ui u : S)
		{
			SR[S_end] = u;
			SR_rid[u] = S_end++;
		}

		R_end=S_end;
		for (ui u : C)
		{
			SR[R_end] = u;
			SR_rid[u] = R_end++;
		}

		for (ui i = 0; i < R_end; ++i)
		{
			ui v = SR[i];
			ui dinS = 0, dinR = 0;
			char *row = matrix + (long long)v * n;

			for (ui j = 0; j < S_end; ++j)
				if (row[SR[j]])
					++dinS;

			for (ui j = S_end; j < R_end; ++j)
				if (row[SR[j]])
					++dinR;

			degree_in_S[v] = dinS;
			degree[v] = dinS + dinR;
		}

		return ub;
	}
};
#endif
