/*
 * CTPrune.h
 *
 *  Created on: 31 May 2023
 *      Author: ljchang@outlook.com
 */

#ifndef CTPRUNE_H_
#define CTPRUNE_H_

#include <cstring>
#include <algorithm>
#include <vector>

using ui = unsigned int; // vertex type
using ept = unsigned int; // edge pointer type; unsigned int can be used to process upto two billion undirected edges

namespace CTPrune {
	// orient graph
	void orient_graph(ui n, ept m, ui *peel_sequence, ept *pstart, ept *pend, ui *edges, ui *rid) {
		for(ui i = 0;i < n;i ++) rid[peel_sequence[i]] = i;
		for(ui i = 0;i < n;i ++) {
			ept &end = pend[i] = pstart[i];
			for(ept j = pstart[i];j < pstart[i+1];j ++) if(rid[edges[j]] > rid[i]) edges[end ++] = edges[j];
		}
	}

	// oriented triangle counting
	void oriented_triangle_counting(ui n, ept m, ept *pstart, ept *pend, ui *edges, ui *tri_cnt, ui *adj) {
		memset(adj, 0, sizeof(ui)*n);
		// long long cnt = 0;
		memset(tri_cnt, 0, sizeof(ui)*m);
		for(ui u = 0;u < n;u ++) {
			for(ept j = pstart[u];j < pend[u];j ++) adj[edges[j]] = j+1;

			for(ept j = pstart[u];j < pend[u];j ++) {
				ui v = edges[j];
				for(ept k = pstart[v];k < pend[v];k ++) if(adj[edges[k]]) {
					++ tri_cnt[j];
					++ tri_cnt[k];
					++ tri_cnt[adj[edges[k]]-1];
					// ++ cnt;
				}
			}

			for(ept j = pstart[u];j < pend[u];j ++) adj[edges[j]] = 0;
		}
	}

	bool remove_and_shrink_oriented_tri(ui &n, ept &m, ui triangle_threshold, ui *out_mapping, ui *peel_sequence, ept *pstart, ept *pend, ui *edges, ui *tri_cnt, ui *rid, ui *degree) {
		for(ui i = 0;i < n;i ++) degree[i] = pstart[i+1] - pstart[i];
		ept removed_edges = 0;
		for(ui i = 0;i < n;i ++) for(ui j = pstart[i];j < pend[i];j ++) if(tri_cnt[j] < triangle_threshold) {
			-- degree[i]; -- degree[edges[j]];
			++ removed_edges;
		}

		if(removed_edges <= m/4) return false;

		ui cnt = 0;
		for(ui i = 0;i < n;i ++) if(degree[i] > 0) {
			out_mapping[cnt] = out_mapping[i];
			rid[i] = cnt++;
		}
		ui t_cnt = 0;
		for(ui i = 0;i < n;i ++) if(degree[peel_sequence[i]] > 0) peel_sequence[t_cnt++] = rid[peel_sequence[i]];

		ui pos = 0; cnt = 0;
		for(ui i = 0;i < n;i ++) if(degree[i] > 0) {
			ui start = pstart[i];
			pstart[cnt] = pos;
			for(ui j = start;j < pend[i];j ++) if(tri_cnt[j] >= triangle_threshold) edges[pos ++] = rid[edges[j]];
			pend[cnt] = pos;
			pos = degree[i] + pstart[cnt];
			++ cnt;
		}
		pstart[cnt] = m = pos;
		n = cnt;

		return true;
	}

	// reorganize the adjacency lists
	// and sort each adjacency list to be in increasing order
	void reorganize_oriented_graph(ui n, ui *tri_cnt, ept *pstart, ept *pend, ept *pend2, ui *edges, ept *edges_pointer, ui *buf) {
		ept m = pstart[n];
		std::vector<ept> deg(n, 0), new_pstart(n + 1, 0), outdeg(n, 0), out_pos(n, 0), in_pos(n, 0);
		ept oriented_edges = 0;
		for(ui u = 0; u < n; ++u) {
			outdeg[u] = pend[u] - pstart[u];
			for(ept j = pstart[u]; j < pend[u]; ++j) {
				++deg[u];
				++deg[edges[j]];
				++oriented_edges;
			}
		}
		for(ui i = 0; i < n; ++i) new_pstart[i + 1] = new_pstart[i] + deg[i];
		assert(new_pstart[n] == m);

		std::vector<ui> temp_edges(m), temp_tri(m), temp_buf_edges, temp_buf_tri;
		std::vector<ept> temp_eid(m), temp_buf_eid;
		temp_buf_edges.reserve(m ? (m / n + 2) : 0);
		temp_buf_tri.reserve(m ? (m / n + 2) : 0);
		temp_buf_eid.reserve(m ? (m / n + 2) : 0);

		for(ui i = 0; i < n; ++i) {
			out_pos[i] = new_pstart[i];
			in_pos[i] = new_pstart[i] + outdeg[i];
		}

		ept eid = 0;
		for(ui u = 0; u < n; ++u) {
			for(ept j = pstart[u]; j < pend[u]; ++j) {
				ui v = edges[j];
				ept a = out_pos[u]++;
				ept b = in_pos[v]++;
				temp_edges[a] = v;
				temp_tri[a] = tri_cnt[j];
				temp_eid[a] = eid;
				temp_edges[b] = u;
				temp_tri[b] = tri_cnt[j];
				temp_eid[b] = eid;
				++eid;
			}
		}

		for(ui i = 0; i < n; ++i) {
			ept left = new_pstart[i];
			ept mid = new_pstart[i] + outdeg[i];
			ept right = new_pstart[i + 1];
			if(left == right) continue;
			temp_buf_edges.clear();
			temp_buf_tri.clear();
			temp_buf_eid.clear();
			temp_buf_edges.reserve(right - left);
			temp_buf_tri.reserve(right - left);
			temp_buf_eid.reserve(right - left);
			ept a = left, b = mid;
			while(a < mid || b < right) {
				if(b >= right || (a < mid && temp_edges[a] < temp_edges[b])) {
					temp_buf_edges.push_back(temp_edges[a]);
					temp_buf_tri.push_back(temp_tri[a]);
					temp_buf_eid.push_back(temp_eid[a]);
					++a;
				}
				else {
					temp_buf_edges.push_back(temp_edges[b]);
					temp_buf_tri.push_back(temp_tri[b]);
					temp_buf_eid.push_back(temp_eid[b]);
					++b;
				}
			}
			for(ept off = 0; off < right - left; ++off) {
				edges[left + off] = temp_buf_edges[off];
				tri_cnt[left + off] = temp_buf_tri[off];
				edges_pointer[left + off] = temp_buf_eid[off];
			}
		}

		std::vector<ept> eid_first(eid, m), eid_second(eid, m);
		for(ept idx = 0; idx < m; ++idx) {
			ept cur = edges_pointer[idx];
			if(eid_first[cur] == m) eid_first[cur] = idx;
			else eid_second[cur] = idx;
		}
		for(ept cur = 0; cur < eid; ++cur) {
			assert(eid_first[cur] < m && eid_second[cur] < m);
			edges_pointer[eid_first[cur]] = eid_second[cur];
			edges_pointer[eid_second[cur]] = eid_first[cur];
		}
		for(ui i = 0; i <= n; ++i) pstart[i] = new_pstart[i];
		for(ui i = 0; i < n; ++i) pend[i] = pstart[i + 1];
	}

	bool find(ui w, ept b, ept e, char *deleted, ept &idx, ui *edges) {
		if(b >= e) return false;

		while(b+1 < e) {
			idx = b + (e-b)/2;
			if(edges[idx] > w) e = idx;
			else b = idx;
		}

		idx = b;
		if(edges[idx] == w&&!deleted[idx]) return true;
		return false;
	}

	void compact_neighbors(ui u, ui *tri_cnt, ui *edges_pointer, char *deleted, ept *pstart, ept *pend, ui *edges) {
		ui end = pstart[u];
		for(ui i = pstart[u];i < pend[u];i ++) if(!deleted[i]) {
			edges[end] = edges[i];
			tri_cnt[end] = tri_cnt[i];
			edges_pointer[end] = edges_pointer[i];
			edges_pointer[edges_pointer[end]] = end;
			deleted[end] = 0;
			++ end;
		}
		pend[u] = end;
	}

	void truss_peeling(ui degree_threshold, ui *Qv, ui &Qv_n, ui triangle_threshold, ui *Qe, ept Qe_n, ui *tri_cnt, ept *edges_pointer, char *deleted, ui *degree, ept *pstart, ept *pend, ui *edges) {
		for(ept j = 0;j < Qe_n;j += 2) {
			ui u = Qe[j], v = Qe[j+1], idx;
			find(v, pstart[u], pend[u], deleted, idx, edges);

			ui tri_n = tri_cnt[idx];
			deleted[idx] = deleted[edges_pointer[idx]] = 1;
			if(Qv != nullptr) {
				if(degree[u] == degree_threshold) Qv[Qv_n++] = u;
				if(degree[v] == degree_threshold) Qv[Qv_n++] = v;
			}
			-- degree[u]; -- degree[v];
			if(pend[u]-pstart[u] > degree[u]*2) compact_neighbors(u, tri_cnt, edges_pointer, deleted, pstart, pend, edges);
			if(pend[v]-pstart[v] > degree[v]*2) compact_neighbors(v, tri_cnt, edges_pointer, deleted, pstart, pend, edges);
			if(pend[u]-pstart[u] < pend[v]-pstart[v]) std::swap(u,v);

			if(pend[u]-pstart[u] > (pend[v]-pstart[v])*2) { // binary search
				for(ept k = pstart[v];k < pend[v];k ++) if(!deleted[k]) {
					if(tri_n&&find(edges[k], pstart[u], pend[u], deleted, idx, edges)) {
						-- tri_n;
						-- tri_cnt[edges_pointer[idx]];
						if( (tri_cnt[idx]--) == triangle_threshold) {
							Qe[Qe_n++] = u; Qe[Qe_n++] = edges[idx];
						}
						-- tri_cnt[edges_pointer[k]];
						if( (tri_cnt[k]--) == triangle_threshold) {
							Qe[Qe_n++] = v; Qe[Qe_n++] = edges[k];
						}
					}
				}
			}
			else { // sorted_merge
				ept ii = pstart[u], jj = pstart[v];
				while(ii < pend[u]&&jj < pend[v]) {
					if(edges[ii] == edges[jj]) {
						if(!deleted[ii]&&!deleted[jj]) {
							-- tri_n;
							-- tri_cnt[edges_pointer[ii]];
							if( (tri_cnt[ii]--) == triangle_threshold) {
								Qe[Qe_n++] = u; Qe[Qe_n++] = edges[ii];
							}
							-- tri_cnt[edges_pointer[jj]];
							if( (tri_cnt[jj]--) == triangle_threshold) {
								Qe[Qe_n++] = v; Qe[Qe_n++] = edges[jj];
							}
						}

						++ ii;
						++ jj;
					}
					else if(edges[ii] < edges[jj]) ++ ii;
					else ++ jj;
				}
			}
		}
	}

	void core_truss_copeeling(ui n, ui degree_threshold, ui triangle_threshold, ui *Qv, ui *Qe, ui *tri_cnt, ept *edges_pointer, char *deleted, char *exists, ui *degree, ept *pstart, ept *pend, ui *edges) {
		ept Qe_n = 0;
		for(ui i = 0;i < n;i ++) for(ept j = pstart[i];j < pend[i];j ++) if(tri_cnt[j] < triangle_threshold&&edges[j] > i) {
			Qe[Qe_n++] = i; Qe[Qe_n++] = edges[j];
		}
		ui Qv_n = 0;
		for(ui i = 0;i < n;i ++) if(degree[i] < degree_threshold) Qv[Qv_n++] = i;
		while(Qe_n || Qv_n) {
			while(Qe_n == 0&&Qv_n != 0) {
				ui u = Qv[-- Qv_n]; // delete u from the graph due to have a degree < degree_threshold
				if(degree[u] == 0) continue;

				if(pend[u]-pstart[u] != degree[u]) compact_neighbors(u, tri_cnt, edges_pointer, deleted, pstart, pend, edges);
				assert(pend[u] - pstart[u] == degree[u]);

				for(ept i = pstart[u];i < pend[u];i ++) deleted[i] = deleted[edges_pointer[i]] = exists[edges[i]] = 1;
				degree[u] = 0;

				for(ept i = pstart[u];i < pend[u];i ++) {
					ui v = edges[i];
					if( (degree[v]--) == degree_threshold) Qv[Qv_n++] = v;
					if(pend[v]-pstart[v] > degree[v]*2) compact_neighbors(v, tri_cnt, edges_pointer, deleted, pstart, pend, edges);

					for(ept j = pstart[v];j < pend[v];j ++) if(!deleted[j]&&edges[j] > v&&exists[edges[j]]) {
						-- tri_cnt[edges_pointer[j]];
						if( (tri_cnt[j]--) == triangle_threshold) {
							Qe[Qe_n++] = v, Qe[Qe_n++] = edges[j];
						}
					}
				}

				for(ept i = pstart[u];i < pend[u];i ++) exists[edges[i]] = 0;
			}
			truss_peeling(degree_threshold, Qv, Qv_n, triangle_threshold, Qe, Qe_n, tri_cnt, edges_pointer, deleted, degree, pstart, pend, edges);
			Qe_n = 0;
		}
	}

	// reduce the graph to its maximal subgraph with and minimum edge triangle count at least triangle_threshold
	void truss_pruning(ui &n, ept &m, ui triangle_threshold, ui *peel_sequence, ui *out_mapping, ui *rid, ept *pstart, ui *edges, ui *degree, bool output) {
		if(triangle_threshold == 0) {
			printf("!!! Triangle_threshold is 0\n");
			return ;
		}

		ept *pend = new ui[n+1];
		orient_graph(n, m, peel_sequence, pstart, pend, edges, rid);
		ui *tri_cnt = new ui[m];
		oriented_triangle_counting(n, m, pstart, pend, edges, tri_cnt, rid);

		while(n&&remove_and_shrink_oriented_tri(n, m, triangle_threshold, out_mapping, peel_sequence, pstart, pend, edges, tri_cnt, rid, degree)) {
			oriented_triangle_counting(n, m, pstart, pend, edges, tri_cnt, rid);
		}

		ept *pend_buf = new ept[n+1];
		ept *edges_pointer = new ept[m];
		reorganize_oriented_graph(n, tri_cnt, pstart, pend, pend_buf, edges, edges_pointer, rid);
		delete[] pend_buf; pend_buf = nullptr;

		for(ui i = 0;i < n;i ++) {
			pend[i] = pstart[i+1];
			degree[i] = pstart[i+1] - pstart[i];
		}
		ui *Qe = new ui[m];
		char *deleted = new char[m];
		memset(deleted, 0, sizeof(char)*m);
		ept Qe_n = 0;
		for(ui i = 0;i < n;i ++) for(ui j = pstart[i];j < pend[i];j ++) if(tri_cnt[j] < triangle_threshold&&edges[j] > i) {
			Qe[Qe_n++] = i; Qe[Qe_n++] = edges[j];
		}
		truss_peeling(0, nullptr, Qe_n, triangle_threshold, Qe, Qe_n, tri_cnt, edges_pointer, deleted, degree, pstart, pend, edges);

		ui cnt = 0;
		for(ui i = 0;i < n;i ++) if(degree[i] > 0) {
			out_mapping[cnt] = out_mapping[i];
			rid[i] = cnt++;
		}
		ui t_cnt = 0;
		for(ui i = 0;i < n;i ++) if(degree[peel_sequence[i]] > 0) peel_sequence[t_cnt++] = rid[peel_sequence[i]];
		ui pos = 0; cnt = 0;
		for(ui i = 0;i < n;i ++) if(degree[i] > 0) {
			ui start = pstart[i];
			pstart[cnt] = pos;
			for(ui j = start;j < pend[i];j ++) if(!deleted[j]) edges[pos++] = rid[edges[j]];
			++ cnt;
		}
		pstart[cnt] = m = pos;
		n = cnt;

		delete[] Qe;
		delete[] deleted;
		delete[] edges_pointer;
		delete[] pend;
		delete[] tri_cnt;

		if(output) printf("*** After truss_pruning: n = %u, m = %lu (undirected)\n", n, m/2);
	}

	// reduce the graph to its maximal subgraph with minimum degree at least degree_threshold and minimum edge triangle count at least triangle_threshold
	void core_truss_copruning(ui &n, ept &m, ui degree_threshold, ui triangle_threshold, ui *peel_sequence, ui *out_mapping, ui *rid, ept *pstart, ui *edges, ui *degree, bool output) {
		if(triangle_threshold == 0) {
			printf("!!! Triangle_threshold is 0\n");
			return ;
		}
		if(degree_threshold <= triangle_threshold+1) {
			printf("!!! Degree_threshold <= triangle_threshold + 1, please invoke truss_pruning\n");
			return ;
		}

		ept *pend = new ui[n+1];
		orient_graph(n, m, peel_sequence, pstart, pend, edges, rid);
		ui *tri_cnt = new ui[m];
		oriented_triangle_counting(n, m, pstart, pend, edges, tri_cnt, rid);

		while(n&&remove_and_shrink_oriented_tri(n, m, triangle_threshold, out_mapping, peel_sequence, pstart, pend, edges, tri_cnt, rid, degree)) {
			oriented_triangle_counting(n, m, pstart, pend, edges, tri_cnt, rid);
		}

		ept *pend_buf = new ept[n+1];
		ept *edges_pointer = new ept[m];
		reorganize_oriented_graph(n, tri_cnt, pstart, pend, pend_buf, edges, edges_pointer, rid);
		delete[] pend_buf; pend_buf = nullptr;

		for(ui i = 0;i < n;i ++) {
			pend[i] = pstart[i+1];
			degree[i] = pstart[i+1] - pstart[i];
		}
		ui *Qe = new ui[m];
		ui *Qv = new ui[n];
		char *deleted = new char[m];
		memset(deleted, 0, sizeof(char)*m);
		char *exists = new char[n];
		memset(exists, 0, sizeof(char)*n);
		core_truss_copeeling(n, degree_threshold, triangle_threshold, Qv, Qe, tri_cnt, edges_pointer, deleted, exists, degree, pstart, pend, edges);

		ui cnt = 0;
		for(ui i = 0;i < n;i ++) if(degree[i] > 0) {
			out_mapping[cnt] = out_mapping[i];
			rid[i] = cnt++;
		}
		ui t_cnt = 0;
		for(ui i = 0;i < n;i ++) if(degree[peel_sequence[i]] > 0) peel_sequence[t_cnt++] = rid[peel_sequence[i]];
		ui pos = 0; cnt = 0;
		for(ui i = 0;i < n;i ++) if(degree[i] > 0) {
			ui start = pstart[i];
			pstart[cnt] = pos;
			for(ui j = start;j < pend[i];j ++) if(!deleted[j]) edges[pos++] = rid[edges[j]];
			degree[cnt]=degree[i];
			++ cnt;
		}
		pstart[cnt] = m = pos;
		n = cnt;

		delete[] Qv;
		delete[] Qe;
		delete[] deleted;
		delete[] edges_pointer;
		delete[] pend;
		delete[] tri_cnt;

		if(output) printf("*** After core_truss_copruning: n = %u, m = %lu (undirected)\n", n, m/2);
	}

	void check_core_pruning(ui n, ui m, ui degree_threshold, ept* pstart, ui *edges) {
		for(ui i = 0;i < n;i ++) {
			if(pstart[i+1]-pstart[i] < degree_threshold) printf("!!! WA degree in check_core_pruning\n");
			for(ept j = pstart[i];j < pstart[i+1];j ++) if(edges[j] >= n) printf("!!! WA edge in check_core_pruning\n");
		}
	}

	void check_truss_pruning(ui n, ui m, ui triangle_threshold, ept* pstart, ui *edges, char *exists) {
		memset(exists, 0, sizeof(char)*n);
		for(ui i = 0;i < n;i ++) {
			for(ept j = pstart[i];j < pstart[i+1];j ++) exists[edges[j]] = 1;
			for(ept j = pstart[i];j < pstart[i+1];j ++) {
				ui v = edges[j], cn = 0;
				for(ept k = pstart[v];k < pstart[v+1];k ++) if(exists[edges[k]]) ++ cn;
				if(cn < triangle_threshold) printf("!!! WA triangle in check_truss_pruning\n");
			}
			for(ept j = pstart[i];j < pstart[i+1];j ++) exists[edges[j]] = 0;
		}
	}
}

#endif /* CTPRUNE_H_ */
