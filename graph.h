#include <vector>

class Graph {
public:
	Graph(const int maxn = 128): n(maxn+2), S(maxn), T(maxn+1), sumEdge(2), flowS(0) {
		son.resize(maxn+5);
		d.resize(maxn+5);
		q.resize(maxn+5);
		pre.resize(maxn+5);
		point.push_back(0);point.push_back(0);
		next.push_back(0);next.push_back(0);
		len.push_back(0);len.push_back(0);
	}
	inline int getS() {return S;}
	inline int getT() {return T;}
	void add_edge(const int a, const int b, const double d1, const double d2) {
		point.push_back(b);
		next.push_back(son[a]);
		len.push_back(d1);
		son[a] = sumEdge++;
		point.push_back(a);
		next.push_back(son[b]);
		len.push_back(d2);
		son[b] = sumEdge++;
	}
	bool extended_path() {
		for (int i=0; i < n; i++) d[i] = 0;
		q[0]=S; d[S]=1; 
		for (int head=-1,tail=0; head++<tail;) {
			for (int u = son[q[head]]; u; u = next[u])
				if (len[u] && !d[point[u]]) {
					d[point[u]] = d[q[head]] + 1;
					q[++tail] = point[u];
				}
			if (d[T]) return true;
		}
		return false;
	}
	void push_flow() {
		double flow = 1 << 15;
		for (int v=T; v!=S; v=pre[v])
			flow = flow<len[preE[v]] ? flow:len[preE[v]];
		for (int v=T; v!=S; v=pre[v]) {
			len[preE[v]] -= flow;
			len[preE[v]^1] += flow;
		}
		flowS += flow;
	}
	void dinic(int u) {
		if (u!=T) {
			for (int v=son[u]; v; v = next[v]) {
				int goa = point[v];
				if (len[v] && d[goa] == d[u] + 1) {
					pre[goa] = u; preE[goa] = v;
					dinic(goa);
				}
			}
			d[u]=0;
		} else push_flow();
	}
	double maxflow() {
		pre.resize(sumEdge); preE.resize(sumEdge);
		for (; extended_path(); dinic(S));
		return flowS;
	}
	bool check_type(int u) {
		return d[u] > 0;
	}

	int n, S, T, sumEdge;
	std::vector<int> son, next, point, d, q, pre, preE;
	std::vector<double> len;
	double flowS;
};