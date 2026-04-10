#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <limits>
#include <queue>
#include <cmath>
#include <set>
#include <unordered_set>
#include <random>
#include <sstream>

using namespace std;

const int INF = 1e9;

// Hungarian Algorithm for minimum weight perfect matching
int hungarian(const vector<vector<int>>& coeff) {
    if (coeff.empty()) return 0;
    int n = coeff.size();
    vector<int> u(n + 1, 0), v(n + 1, 0), p(n + 1, 0), way(n + 1, 0);
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        vector<int> minv(n + 1, INF);
        vector<char> used(n + 1, false);
        do {
            used[j0] = true;
            int i0 = p[j0], delta = INF, j1 = 0;
            for (int j = 1; j <= n; ++j) {
                if (!used[j]) {
                    int cur = coeff[i0 - 1][j - 1] - u[i0] - v[j];
                    if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                    if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else minv[j] -= delta;
            }
            j0 = j1;
        } while (p[j0] != 0);
        do { int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0 != 0);
    }
    int res = 0;
    for(int j=1; j<=n; ++j) if(p[j] != 0) res += coeff[p[j]-1][j-1];
    return res;
}

int calculate_cost(const vector<int>& p, const vector<vector<int>>& F, const vector<vector<int>>& D) {
    int cost = 0;
    int n = p.size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cost += F[i][j] * D[p[i]][p[j]];
        }
    }
    return cost;
}

// Global state for B&B
long long nodes_explored = 0;
int best_ub = INF;
vector<int> best_pi;
auto start_time = chrono::high_resolution_clock::now();
double timeout_sec = 60.0;

bool timelimit_reached() {
    auto now = chrono::high_resolution_clock::now();
    double elaps = chrono::duration<double>(now - start_time).count();
    return elaps >= timeout_sec;
}

int compute_glb(const vector<int>& fixed_f, const vector<int>& fixed_l, const vector<vector<int>>& F, const vector<vector<int>>& D, int n) {
    vector<bool> free_f_bool(n, true), free_l_bool(n, true);
    for (int i=0; i<fixed_f.size(); i++) {
        free_f_bool[fixed_f[i]] = false;
        free_l_bool[fixed_l[i]] = false;
    }
    vector<int> free_f, free_l;
    for (int i=0; i<n; i++) {
        if(free_f_bool[i]) free_f.push_back(i);
        if(free_l_bool[i]) free_l.push_back(i);
    }
    
    int c_fixed = 0;
    for(size_t i=0; i<fixed_f.size(); i++) {
        for(size_t j=0; j<fixed_f.size(); j++) {
            c_fixed += F[fixed_f[i]][fixed_f[j]] * D[fixed_l[i]][fixed_l[j]];
        }
    }
    
    if (free_f.empty()) return c_fixed;

    int m = free_f.size();
    vector<vector<int>> M(m, vector<int>(m, 0));
    
    for(int i=0; i<m; i++) {
        int fi = free_f[i];
        vector<int> f_row;
        for(int x : free_f) f_row.push_back(F[fi][x]);
        sort(f_row.begin(), f_row.end());
        
        for(int j=0; j<m; j++) {
            int lj = free_l[j];
            int c1 = 0;
            // fixed to current free
            for(size_t k=0; k<fixed_f.size(); k++) {
                c1 += F[fi][fixed_f[k]] * D[lj][fixed_l[k]];
                c1 += F[fixed_f[k]][fi] * D[fixed_l[k]][lj];
            }
            
            vector<int> d_row;
            for(int y : free_l) d_row.push_back(D[lj][y]);
            sort(d_row.rbegin(), d_row.rend());
            
            int c2 = 0;
            for(int k=0; k<m; k++) c2 += f_row[k] * d_row[k];
            
            M[i][j] = c1 + c2;
        }
    }
    
    return c_fixed + hungarian(M);
}

void bb_base_solve(vector<int>& fixed_f, vector<int>& fixed_l, const vector<vector<int>>& F, const vector<vector<int>>& D, int n) {
    if (timelimit_reached()) return;
    nodes_explored++;
    
    int m = fixed_f.size();
    if (m == n) {
        int c = compute_glb(fixed_f, fixed_l, F, D, n);
        if (c < best_ub) {
            best_ub = c;
            best_pi.resize(n);
            for(int i=0; i<n; i++) best_pi[fixed_f[i]] = fixed_l[i];
        }
        return;
    }
    
    int lb = compute_glb(fixed_f, fixed_l, F, D, n);
    if (lb >= best_ub) return;

    // Next free facility arbitrarily
    vector<bool> used_f(n, false), used_l(n, false);
    for(int x : fixed_f) used_f[x] = true;
    for(int x : fixed_l) used_l[x] = true;
    
    int next_f = -1;
    for(int i=0; i<n; i++) if(!used_f[i]) { next_f = i; break; }
    
    for(int j=0; j<n; j++) {
        if (!used_l[j]) {
            fixed_f.push_back(next_f);
            fixed_l.push_back(j);
            bb_base_solve(fixed_f, fixed_l, F, D, n);
            fixed_f.pop_back();
            fixed_l.pop_back();
        }
    }
}

// B&B Novel: LB-ordered DFS + Max-Min Branching + TS Warmstart
void bb_novel_solve(vector<int>& fixed_f, vector<int>& fixed_l, const vector<vector<int>>& F, const vector<vector<int>>& D, int n) {
    if (timelimit_reached()) return;
    nodes_explored++;
    
    int m = fixed_f.size();
    if (m == n) {
        int c = compute_glb(fixed_f, fixed_l, F, D, n);
        if (c < best_ub) {
            best_ub = c;
            best_pi.resize(n);
            for(int i=0; i<n; i++) best_pi[fixed_f[i]] = fixed_l[i];
        }
        return;
    }
    
    int lb = compute_glb(fixed_f, fixed_l, F, D, n);
    if (lb >= best_ub) return;

    vector<bool> used_f(n, false), used_l(n, false);
    for(int x : fixed_f) used_f[x] = true;
    for(int x : fixed_l) used_l[x] = true;
    
    // Max-min branching: find free_f that maximizes min_lb over all branches
    int next_f = -1;
    int max_min_lb = -1;
    vector<pair<int, int>> best_branches;

    for (int i = 0; i < n; i++) {
        if (used_f[i]) continue;
        int min_lb = INF;
        vector<pair<int, int>> current_branches; // pair<child_lb, location>
        
        for (int j = 0; j < n; j++) {
            if (used_l[j]) continue;
            fixed_f.push_back(i);
            fixed_l.push_back(j);
            int child_lb = compute_glb(fixed_f, fixed_l, F, D, n);
            current_branches.push_back({child_lb, j});
            min_lb = min(min_lb, child_lb);
            fixed_f.pop_back();
            fixed_l.pop_back();
        }
        if (min_lb > max_min_lb) {
            max_min_lb = min_lb;
            next_f = i;
            best_branches = current_branches;
        }
    }
    
    if (max_min_lb >= best_ub) return; // Pruned early by lookahead
    
    // Sort branches by LB ascending so we explore most promising first
    sort(best_branches.begin(), best_branches.end());
    
    for (auto& branch : best_branches) {
        int j = branch.second;
        fixed_f.push_back(next_f);
        fixed_l.push_back(j);
        bb_novel_solve(fixed_f, fixed_l, F, D, n);
        fixed_f.pop_back();
        fixed_l.pop_back();
    }
}

mt19937 rng(42);

// Fast Incremental Delta Evaluation
int evaluate_swap_delta(int r, int s, const vector<int>& pi, const vector<vector<int>>& F, const vector<vector<int>>& D, int n) {
    int p = pi[r];
    int q = pi[s];
    int delta = 0;
    for (int k = 0; k < n; k++) {
        if (k != r && k != s) {
            delta += (F[r][k] - F[s][k]) * (D[q][pi[k]] - D[p][pi[k]]) + 
                     (F[k][r] - F[k][s]) * (D[pi[k]][q] - D[pi[k]][p]);
        }
    }
    delta += (F[r][s] - F[s][r]) * (D[q][p] - D[p][q]);
    return delta;
}

// TS Base: Static Tenure
void ts_base(const vector<vector<int>>& F, const vector<vector<int>>& D, int n, int max_iter) {
    if(n == 0) return;
    vector<int> pi(n);
    iota(pi.begin(), pi.end(), 0);
    shuffle(pi.begin(), pi.end(), rng);
    
    int current_cost = calculate_cost(pi, F, D);
    int best_cost = current_cost;
    best_pi = pi;
    
    int tenure = n / 4;
    vector<vector<int>> tabu_list(n, vector<int>(n, 0));
    
    for (int iter = 1; iter <= max_iter; iter++) {
        int best_delta = INF;
        int best_r = -1, best_s = -1;
        
        for (int r = 0; r < n - 1; r++) {
            for (int s = r + 1; s < n; s++) {
                int delta = evaluate_swap_delta(r, s, pi, F, D, n);
                bool is_tabu = tabu_list[r][s] >= iter;
                bool aspiration = (current_cost + delta < best_cost);
                
                if ((!is_tabu || aspiration) && delta < best_delta) {
                    best_delta = delta;
                    best_r = r;
                    best_s = s;
                }
            }
        }
        
        if (best_r == -1) break; // Trapped (rare)
        
        swap(pi[best_r], pi[best_s]);
        current_cost += best_delta;
        tabu_list[best_r][best_s] = iter + tenure;
        
        if (current_cost < best_cost) {
            best_cost = current_cost;
            best_pi = pi;
        }
        if (timelimit_reached()) break;
    }
    
    best_ub = best_cost;
}

// Path Relinking: finds the best intermediate solution between two permutations
vector<int> path_relink(vector<int> start_pi, const vector<int>& target_pi, const vector<vector<int>>& F, const vector<vector<int>>& D, int n) {
    vector<int> curr_pi = start_pi;
    vector<int> best_intermediate = start_pi;
    int curr_cost = calculate_cost(curr_pi, F, D);
    int best_cost = curr_cost;
    
    while (curr_pi != target_pi) {
        int best_move_delta = INF;
        int best_r = -1, best_s = -1;
        
        for (int i = 0; i < n; i++) {
            if (curr_pi[i] != target_pi[i]) {
                // Find where target_pi[i] is in curr_pi
                int j = i + 1;
                for (; j < n; j++) if(curr_pi[j] == target_pi[i]) break;
                if(j < n) {
                    int delta = evaluate_swap_delta(i, j, curr_pi, F, D, n);
                    if (delta < best_move_delta) {
                        best_move_delta = delta;
                        best_r = i;
                        best_s = j;
                    }
                }
            }
        }
        if (best_r == -1) break;
        swap(curr_pi[best_r], curr_pi[best_s]);
        curr_cost += best_move_delta;
        if (curr_cost < best_cost) {
            best_cost = curr_cost;
            best_intermediate = curr_pi;
        }
    }
    return best_intermediate;
}

// TS Novel: Reactive Tenure + Elite Pool + Path Relinking
void ts_novel(const vector<vector<int>>& F, const vector<vector<int>>& D, int n, int max_iter) {
    if(n == 0) return;
    vector<int> pi(n);
    iota(pi.begin(), pi.end(), 0);
    shuffle(pi.begin(), pi.end(), rng);
    
    int current_cost = calculate_cost(pi, F, D);
    int best_cost = current_cost;
    best_pi = pi;
    
    double tenure = n / 4.0;
    vector<vector<int>> tabu_list(n, vector<int>(n, 0));
    set<long long> visited_hashes;
    
    // Elite pool to store diverse best solutions
    vector<pair<int, vector<int>>> elite_pool;
    int no_improve = 0;
    
    auto hash_pi = [&](const vector<int>& p) {
        long long h = 0;
        for(int x : p) h = (h * 31 + x) % 1000000009;
        return h;
    };
    
    for (int iter = 1; iter <= max_iter; iter++) {
        visited_hashes.insert(hash_pi(pi));
        
        int best_delta = INF;
        int best_r = -1, best_s = -1;
        
        for (int r = 0; r < n - 1; r++) {
            for (int s = r + 1; s < n; s++) {
                int delta = evaluate_swap_delta(r, s, pi, F, D, n);
                bool is_tabu = tabu_list[r][s] >= iter;
                bool aspiration = (current_cost + delta < best_cost);
                
                if ((!is_tabu || aspiration) && delta < best_delta) {
                    best_delta = delta;
                    best_r = r;
                    best_s = s;
                }
            }
        }
        
        if (best_r == -1) break;
        
        swap(pi[best_r], pi[best_s]);
        current_cost += best_delta;
        tabu_list[best_r][best_s] = iter + (int)tenure;
        
        if (visited_hashes.count(hash_pi(pi))) tenure = min((double)n, tenure * 1.1);
        else tenure = max(1.0, tenure * 0.995);
        
        if (current_cost < best_cost) {
            best_cost = current_cost;
            best_pi = pi;
            no_improve = 0;
            
            // Manage elite pool
            elite_pool.push_back({best_cost, best_pi});
            sort(elite_pool.begin(), elite_pool.end());
            if(elite_pool.size() > 5) elite_pool.pop_back();
        } else {
            no_improve++;
        }
        
        // Path Relinking Diversification
        if (no_improve > 10 * n && elite_pool.size() >= 2) {
            int idx1 = rng() % elite_pool.size();
            int idx2 = rng() % elite_pool.size();
            while(idx1 == idx2) idx2 = rng() % elite_pool.size();
            
            pi = path_relink(elite_pool[idx1].second, elite_pool[idx2].second, F, D, n);
            current_cost = calculate_cost(pi, F, D);
            no_improve = 0;
            fill(tabu_list.begin(), tabu_list.end(), vector<int>(n, 0)); // Reset tabu
        }
        
        if (timelimit_reached()) break;
    }
    best_ub = best_cost;
}

double get_peak_memory_mb() {
    ifstream f("/proc/self/status");
    string line;
    while(getline(f, line)) {
        if(line.substr(0, 6) == "VmHWM:") {
            stringstream ss(line.substr(6));
            double mem; string unit;
            ss >> mem >> unit;
            return mem / 1024.0; 
        }
    }
    return 0.0;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        cerr << "Usage: ./qap_solver <file.dat> <algorithm> <timeout/max_iter>" << endl;
        return 1;
    }
    
    string file = argv[1];
    string algo = argv[2];
    timeout_sec = stod(argv[3]);
    int max_iter = stoi(argv[3]);
    
    ifstream fin(file);
    if (!fin) { cerr << "Could not open file " << file << endl; return 1; }
    
    int n; fin >> n;
    vector<vector<int>> F(n, vector<int>(n));
    vector<vector<int>> D(n, vector<int>(n));
    for (int i=0; i<n; i++) for(int j=0; j<n; j++) fin >> F[i][j];
    for (int i=0; i<n; i++) for(int j=0; j<n; j++) fin >> D[i][j];
    
    start_time = chrono::high_resolution_clock::now();
    nodes_explored = 0;
    best_ub = INF;
    
    if (algo == "bb_base") {
        vector<int> fixed_f, fixed_l;
        bb_base_solve(fixed_f, fixed_l, F, D, n);
    } 
    else if (algo == "bb_novel") {
        // Quick TS warm-start to get initial UB
        double original_timeout = timeout_sec;
        timeout_sec = min(timeout_sec, 0.2); // spend max 0.2s on TS warmstart
        ts_base(F, D, n, 100); 
        timeout_sec = original_timeout;
        
        vector<int> fixed_f, fixed_l;
        bb_novel_solve(fixed_f, fixed_l, F, D, n);
    } 
    else if (algo == "ts_base") {
        ts_base(F, D, n, max_iter);
    } 
    else if (algo == "ts_novel") {
        ts_novel(F, D, n, max_iter);
    }
    else {
        cerr << "Unknown algorithm." << endl;
        return 1;
    }
    
    auto end_time = chrono::high_resolution_clock::now();
    double time_taken = chrono::duration<double>(end_time - start_time).count();
    
    // cost, time, nodes/iters, memory
    cout << best_ub << "," << time_taken << "," << nodes_explored << "," << get_peak_memory_mb() << endl;
    return 0;
}
