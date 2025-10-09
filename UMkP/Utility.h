#ifndef _UTILITY_H_
#define _UTILITY_H_

#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <queue>
#include <set>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <map>
#include <set>
#include <cmath>
#include <queue>
#include <stack>
#include <cstring>
#include <iomanip>
#include <limits>
#include <cassert>
#include <numeric>
#include <fstream>
#include <sstream>
#include <unordered_map>
using namespace std;

#include <sys/time.h> // gettimeofday
#include <unistd.h>
#include <chrono>
#define NDEBUG
#include <cassert>

// #define NAIVE

#ifndef NAIVE
#define MY_SOLVER
#endif

using ui = unsigned int; // vertex type
using ept = unsigned int; // edge pointer type; unsigned int can be used to process upto two billion undirected edges

#define pb push_back
#define mp make_pair
using namespace std;
class Utility {
public:
	static FILE *open_file(const char *file_name, const char *mode) {
		FILE *f = fopen(file_name, mode);
		if(f == nullptr) {
			printf("Can not open file: %s\n", file_name);
			exit(1);
		}

		return f;
	}

	static std::string integer_to_string(long long number) {
		std::vector<ui> sequence;
		if(number == 0) sequence.pb(0);
		while(number > 0) {
			sequence.pb(number%1000);
			number /= 1000;
		}

		char buf[5];
		std::string res;
		for(ui i = sequence.size();i > 0;i --) {
			if(i == sequence.size()) sprintf(buf, "%u", sequence[i-1]);
			else sprintf(buf, ",%03u", sequence[i-1]);
			res += std::string(buf);
		}
		return res;
	}
};

class MBitSet
{
private:
public:
	ui n;
	ui cap;
	ui *buf;

	MBitSet()
	{
		buf = nullptr;
		cap = n = 0;
	}
	MBitSet(ui _cap)
	{
		cap = _cap;
		n = (cap >> 5) + 1;
		buf = new ui[n];
		fill(buf, buf + n, 0);
		// for (ui i = 0; i < n; ++i)
		// 	buf[i] = 0;
	}
	void init(ui _cap)
	{
		cap = _cap;
		n = (cap >> 5) + 1;
		buf = new ui[n];
		fill(buf, buf + n, 0);
	}
	~MBitSet()
	{
		// todo doing double free, see what causing it and fix

		if (buf != nullptr)
			// delete[] buf;
			buf = nullptr;
	}
	void reset()
	{
		fill(buf, buf + n, 0);
	}
	void reset(ui n)
	{
		fill(buf, buf + (n>>5)+1, 0);
	}
	void setAll()
	{
		fill(buf, buf + n, 0xffffffff);
	}
	// FLIP all the bits
	void flip()
	{
		for (ui i = 0; i < n; ++i)
			buf[i] = ~buf[i];
	}
	void set(ui x)
	{
		// assert(x < cap);
		buf[x >> 5] |= (ui)1 << (x & 31);
	}

	bool test(ui x)
	{
		// cout << x << " " << n << " " << cap << endl;
		return buf[x >> 5] >> (x & 31) & 1;
	}

	bool empty()
	{
		for (ui i = 0; i < n; ++i)
			if (buf[i])
				return false;
		return true;
	}
	void setup(vector<ui> &adj, ui m)
	{
		m = (m >> 5) + 1;
		fill(buf, buf + m, 0);
		
		for (ui u : adj)
			set(u);
		
	}
};
#define x first
#define y second

using namespace std;
using ll = long long;
using ui = unsigned int;
using pii = pair<ui, ui>;

const int INF = 0x3f3f3f3f;

int paramK;
int lb;

double list_triangle_time;

inline ll get_system_time_microsecond()
{
    auto duration = std::chrono::system_clock::now().time_since_epoch();
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration);
    return static_cast<long long>(microseconds.count());
}

string get_file_name(string str)
{
    string ret = "";
    for (char ch : str)
    {
        ret += ch;
        if (ch == '\\' || ch == '/')
            ret = "";
    }
    return ret;
}

string get_file_name_without_suffix(string name)
{
    name = get_file_name(name);
    string ret = "";
    for (char ch : name)
    {
        if (ch == '.')
            break;
        ret += ch;
    }
    return ret;
}

string get_file_name_suffix(string f)
{
    string ret = "";
    bool has_dot = 0;
    for (char ch : f)
    {
        if (ch == '.')
        {
            has_dot = 1;
            ret = "";
        }
        else if (has_dot)
            ret += ch;
    }
    return ret;
}

/**
 * faster than std::sort
 * We first sort using "second(y)", then sort using "first(x)"
 * because the countingSort is stable, the final array is sorted in the ascending order
 */
void countingSort(vector<pii> &a, int k)
{
    ui *cnt = new ui[k];
    // fisrt, we sort using y
    memset(cnt, 0, sizeof(ui) * k);
    for (auto &h : a)
        cnt[h.y]++;
    for (int i = 1; i < k; i++)
        cnt[i] += cnt[i - 1];
    vector<pii> out(a.size());
    for (ll i = (ll)a.size() - 1; i >= 0; i--)
        out[cnt[a[i].y] - 1] = a[i], cnt[a[i].y]--;
    // next, we sort using x
    memset(cnt, 0, sizeof(ui) * k);
    for (auto &h : out)
        cnt[h.x]++;
    for (int i = 1; i < k; i++)
        cnt[i] += cnt[i - 1];
    for (ll i = (ll)a.size() - 1; i >= 0; i--)
        a[cnt[out[i].x] - 1] = out[i], cnt[out[i].x]--;
    delete[] cnt;
}

void unique_pii(vector<pii> &a, int n)
{
    countingSort(a, n);
    a.erase(unique(a.begin(), a.end()), a.end());
}

template <typename T>
void unique_vector(vector<T> &a)
{
    sort(a.begin(), a.end());
    a.erase(unique(a.begin(), a.end()), a.end());
}

template <typename T>
bool has(vector<T> &a, T x)
{
    auto it = lower_bound(a.begin(), a.end(), x);
    return it != a.end() && *it == x;
}

template <typename T>
bool has(T *st, T *ed, T x)
{
    auto it = lower_bound(st, ed, x);
    return it != ed && (*it) == x;
}

template <typename T>
int find(vector<T> &a, T x)
{
    return lower_bound(a.begin(), a.end(), x) - a.begin();
}

template <typename T>
int find(T *st, T *ed, T x)
{
    return lower_bound(st, ed, x) - st;
}

template <typename T>
int intersect_count(T *a, T *b, T *l, T *r) // return [a,b) ∩ [l,r)
{
    int ret = 0;
    while (a < b && l < r)
    {
        if ((*a) == (*l))
            ret++, a++, l++;
        else if ((*a) > (*l))
            l++;
        else
            a++;
    }
    return ret;
}

void print_progress_bar(double percentage, bool enter = false)
{
#ifdef NO_PROGRESS_BAR
    return;
#endif
    const int bar_length = 50;
    const char fill_char = '#';
    const char empty_char = '-';
    int filled_length = (int)(percentage * bar_length);
    int empty_length = bar_length - filled_length;
    char buffer[bar_length + 10];
    sprintf(buffer, "\r%.2lf%% [", percentage * 100);
    int len = strlen(buffer);
    for (int i = 0; i < filled_length; i++)
    {
        buffer[len + i] = fill_char;
    }
    for (int i = 0; i < empty_length; i++)
    {
        buffer[len + filled_length + i] = empty_char;
    }
    buffer[bar_length + len] = ']';
    write(STDOUT_FILENO, buffer, bar_length + len + 1);
    fflush(stdout);
    if (enter)
        cout << endl;
}

template <typename T>
void print_set(set<T> &s)
{
    if (!s.size())
        return;
    auto it = s.begin();
    cout << (*it);
    it++;
    while (it != s.end())
    {
        cout << ", " << (*it);
        it++;
    }
    cout << endl;
}

template <typename T>
int countBits(T x)
{
    int count = 0;
    while (x)
    {
        x &= (x - 1);
        count++;
    }
    return count;
}

void print_time(double start)
{
    printf("use time %.4lf s\n", (get_system_time_microsecond() - start) / 1e6);
    fflush(stdout);
}

void print_module_time(string name, double time)
{
    printf("%s-time: %.4lf s  ", name.c_str(), time / 1e6);
}

ui read()
{
    char ch;
    while ((ch = getchar()) < '0' || ch > '9')
        ;
    ui ret = 0;
    while (ch >= '0' && ch <= '9')
    {
        ret = ret * 10 + ch - '0';
        ch = getchar();
    }
    return ret;
}

class Timer
{
    ll start_time;
    string log;

public:
    Timer()
    {
        start_time = get_system_time_microsecond();
    }
    Timer(string s) : log(s)
    {
        start_time = get_system_time_microsecond();
    }
    double get_time()
    {
        return (get_system_time_microsecond() - start_time);
    }
    void print_time()
    {
        printf("%s use time: %.4lf s\n", log.c_str(), get_time() / 1e6);
        fflush(stdout);
    }
    double get_time_seconds()
    {
        return (get_system_time_microsecond() - start_time) / 1e6;
    }
};

#endif

