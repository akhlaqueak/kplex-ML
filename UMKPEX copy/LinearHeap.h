#ifndef LINEARHEAP_H
#define LINEARHEAP_H

#include "Utility.h"

class ListLinearHeap
{
private:
    ui n;
    ui key_cap;

    ui min_key;
    ui max_key;

    ui *key_s;
    ui *head_s;
    ui *pre_s;
    ui *next_s;

public:
    ListLinearHeap(ui _n, ui _key_cap)
    {
        n = _n;
        key_cap = _key_cap;
        min_key = max_key = key_cap;
        head_s = key_s = pre_s = next_s = nullptr;
    }

    ~ListLinearHeap()
    {
        if (head_s != nullptr)
        {
            delete[] head_s;
            head_s = nullptr;
        }
        if (pre_s != nullptr)
        {
            delete[] pre_s;
            pre_s = nullptr;
        }
        if (next_s != nullptr)
        {
            delete[] next_s;
            next_s = nullptr;
        }
        if (key_s != nullptr)
        {
            delete[] key_s;
            key_s = nullptr;
        }
    }

    void init(ui _n, ui _key_cap, ui *_id_s, ui *_key_s)
    {
        if (key_s == nullptr)
            key_s = new ui[n];
        if (pre_s == nullptr)
            pre_s = new ui[n];
        if (next_s == nullptr)
            next_s = new ui[n];
        if (head_s == nullptr)
            head_s = new ui[key_cap + 1];

        min_key = max_key = _key_cap;
        for (ui i = 0; i <= _key_cap; i++)
            head_s[i] = n;

        for (ui i = 0; i < _n; i++)
        {
            ui id = _id_s[i];
            ui key = _key_s[id];

            key_s[id] = key;
            pre_s[id] = n;
            next_s[id] = head_s[key];
            if (head_s[key] != n)
                pre_s[head_s[key]] = id;
            head_s[key] = id;

            if (key < min_key)
                min_key = key;
        }
    }

    bool pop_min(ui &id, ui &key)
    {
        while (min_key <= max_key && head_s[min_key] == n)
            ++min_key;
        if (min_key > max_key)
            return false;

        id = head_s[min_key];
        key = min_key;

        key_s[id] = key_cap + 1;
        head_s[min_key] = next_s[id];
        if (head_s[min_key] != n)
            pre_s[head_s[min_key]] = n;
        return true;
    }

    ui decrement(ui id, ui dec)
    {
        if (key_s[id] > key_cap)
            return 0;

        if (pre_s[id] == n)
        {
            head_s[key_s[id]] = next_s[id];
            if (next_s[id] != n)
                pre_s[next_s[id]] = n;
        }
        else
        {
            ui pid = pre_s[id];
            next_s[pid] = next_s[id];
            if (next_s[id] != n)
                pre_s[next_s[id]] = pid;
        }

        ui &key = key_s[id];
        key -= dec;
        pre_s[id] = n;
        next_s[id] = head_s[key];
        if (head_s[key] != n)
            pre_s[head_s[key]] = id;
        head_s[key] = id;

        if (key < min_key)
            min_key = key;
        return key;
    }
};

/**
 * @brief used for degeneracy order, which can be done in O(m).
 * based on linked list
 */
class LinearHeap
{
    struct Node
    {
        ui pre, ne;
    };
    ui *h;
    Node *nodes;

public:
    ui max_range;
    ui min_key;
    ui sz, n; // sz will decrease , but n won't
    LinearHeap() : max_range(0), n(0), sz(0), min_key(0), h(nullptr), nodes(nullptr)
    {
    }
    LinearHeap(ui _range, ui _n) : max_range(_range), n(_n), sz(0), h(nullptr), nodes(nullptr)
    {
        h = new ui[max_range];
        for (ui i = 0; i < max_range; i++)
            h[i] = n;
        nodes = new Node[n];
        // for (ui i = 0; i < n; i++)
        //     nodes[i] = {n, n};
        min_key = max_range;
    }
    LinearHeap(ui _range, ui _n, ui pd[]) : max_range(_range), n(_n), sz(0), h(nullptr), nodes(nullptr)
    {
        h = new ui[max_range];
        for (ui i = 0; i < max_range; i++)
            h[i] = n;
        nodes = new Node[n];
        // for (ui i = 0; i < n; i++)
        //     nodes[i] = {n, n};
        min_key = max_range;
        for (ui i = 0; i < n; i++)
            insert(pd[i], i);
    }
    LinearHeap(ui _range, ui _n, int pd[]) : max_range(_range), n(_n), sz(0), h(nullptr), nodes(nullptr)
    {
        h = new ui[max_range];
        for (ui i = 0; i < max_range; i++)
            h[i] = n;
        nodes = new Node[n];
        // for (ui i = 0; i < n; i++)
        //     nodes[i] = {n, n};
        min_key = max_range;
        for (ui i = 0; i < n; i++)
            insert(pd[i], i);
    }
    LinearHeap(ui _range, ui _n, vector<ui> &pd) : max_range(_range), n(_n), sz(0), h(nullptr), nodes(nullptr)
    {
        h = new ui[max_range];
        for (ui i = 0; i < max_range; i++)
            h[i] = n;
        nodes = new Node[n];
        // for (ui i = 0; i < n; i++)
        //     nodes[i] = {n, n};
        min_key = max_range;
        for (ui i = 0; i < n; i++)
            insert(pd[i], i);
    }
    void init(const vector<ui> &ids, ui pd[])
    {
        for (ui i = 0; i < max_range; i++)
            h[i] = n;
        sz = 0;
        min_key = max_range;
        for (ui id : ids)
            insert(pd[id], id);
    }
    void init(const vector<ui> &ids, int pd[])
    {
        for (ui i = 0; i < max_range; i++)
            h[i] = n;
        sz = 0;
        min_key = max_range;
        for (ui id : ids)
            insert(pd[id], id);
    }
    ~LinearHeap()
    {
        if (h != nullptr)
            delete[] h;
        if (nodes != nullptr)
            delete[] nodes;
    }
    LinearHeap &operator=(const LinearHeap &other)
    {
        max_range = other.max_range;
        n = other.n;
        sz = other.sz;
        min_key = other.min_key;
        if (h != nullptr)
        {
            delete[] h;
        }
        h = new ui[max_range];
        memcpy(h, other.h, sizeof(ui) * max_range);
        if (nodes != nullptr)
        {
            delete[] nodes;
        }
        nodes = new Node[n];
        memcpy(nodes, other.nodes, sizeof(Node) * n);
        return *this;
    }
    // key is the degree of a vertex; val is the index of a vertex
    inline void insert(ui key, ui val)
    {
        min_key = min(min_key, key);
        sz++;
        nodes[val].pre = key + n;
        nodes[val].ne = h[key];
        if (h[key] < n)
            nodes[h[key]].pre = val;
        h[key] = val;
    }
    ui get_min_key()
    {
        assert(sz > 0);
        while (h[min_key] >= n)
            min_key++;
        return min_key;
    }
    inline void delete_node(ui id)
    {
        sz--;
        auto &cur = nodes[id];
        if (cur.pre >= n) // key = cur.pre-n
            h[cur.pre - n] = cur.ne;
        else
            nodes[cur.pre].ne = cur.ne;
        if (cur.ne < n)
            nodes[cur.ne].pre = cur.pre;
    }
    inline void decrease(ui key_now, ui id)
    {
        // delete_node(id);
        // insert(key_now, id);
        int pre = nodes[id].pre, ne = nodes[id].ne;
        if (pre >= n)
        {
            h[pre - n] = ne;
        }
        else
            nodes[pre].ne = ne;
        if (ne < n)
        {
            nodes[ne].pre = pre;
        }
        min_key = min(min_key, key_now);
        ui nxt = h[key_now];
        nodes[id] = {key_now + n, nxt};
        if (nxt < n)
        {
            nodes[nxt].pre = id;
        }
        h[key_now] = id;
    }
    ui get_min_node()
    {
        return h[get_min_key()];
    }
};

#endif
