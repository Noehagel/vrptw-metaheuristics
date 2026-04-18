#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

const double PI = 3.141592653589793;
const double RHO = 6.371e6;
const double OMEGA = 2.0 * PI / 1440.0;

const double DEPOT_LAT = 48.764246;
const double DEPOT_LON = 2.34842;

struct Vehicle {
    int id, cap;
    double rental, fuel, radius, speed;
    double cosc[4], sinc[4];
};

struct Order {
    int id;
    double x, y;
    double weight, start, end, dur;
};

struct Route {
    int v_idx;
    vector<int> ids;
};

struct Instance {
    vector<Vehicle> vehicles;
    vector<Order> orders; // orders[id-1]
    Order depot;
    int N = 0;
    double dists[505][505];
    double eucl[505][505];
};

struct Config {
    string instances_dir = "data/instances";
    string vehicles_file = "data/instances/vehicles.csv";
    string output_dir = "results/solutions";
    int time_limit_sec = 600;
    int seed = 11;
    int first_instance = 1;
    int last_instance = 10;
};

static inline double get_gamma(const Vehicle& v, double t) {
    double g = v.cosc[0];
    for (int n = 1; n < 4; ++n) {
        double ang = n * OMEGA * t;
        g += v.cosc[n] * cos(ang) + v.sinc[n] * sin(ang);
    }
    return g;
}

static inline bool feasible_and_slack(const Instance& inst, const vector<int>& ids, int v_idx, double& min_slack) {
    const auto& v = inst.vehicles[v_idx];
    double t = 0.0, load = 0.0;
    int cur = 0;
    min_slack = 1e18;

    for (int id : ids) {
        const auto& o = inst.orders[id - 1];
        load += o.weight;
        if (load > v.cap + 1e-6) return false;

        t += (inst.dists[cur][id] / v.speed) * get_gamma(v, t);

        if (t < o.start) t = o.start;
        if (t > o.end + 1e-6) return false;

        min_slack = min(min_slack, o.end - t);
        t += o.dur;
        cur = id;
    }
    return true;
}

static inline bool is_feasible(const Instance& inst, const vector<int>& ids, int v_idx) {
    double slack;
    return feasible_and_slack(inst, ids, v_idx, slack);
}

static inline double route_weight_ids(const Instance& inst, const vector<int>& ids) {
    double w = 0.0;
    for (int id : ids) w += inst.orders[id - 1].weight;
    return w;
}

static inline double route_weight(const Instance& inst, const Route& r) {
    return route_weight_ids(inst, r.ids);
}

static inline void candidate_vehicles(const Instance& inst, double required_cap, vector<int>& cand, int topk = 4) {
    cand.clear();
    vector<pair<double, int>> tmp;
    tmp.reserve(inst.vehicles.size());

    for (int i = 0; i < (int)inst.vehicles.size(); ++i) {
        const auto& v = inst.vehicles[i];
        if (v.cap + 1e-6 >= required_cap) {
            double key = v.rental + 1200.0 * v.fuel + 0.01 * v.radius;
            tmp.push_back({key, i});
        }
    }

    sort(tmp.begin(), tmp.end());
    for (int k = 0; k < (int)tmp.size() && k < topk; ++k) cand.push_back(tmp[k].second);
    if (cand.empty()) cand.push_back((int)inst.vehicles.size() - 1);
}

static inline double route_cost_ids_with_vehicle(const Instance& inst, const vector<int>& ids, int v_idx) {
    if (ids.empty()) return 0.0;
    const auto& v = inst.vehicles[v_idx];

    double d = 0.0;
    int prev = 0;
    for (int id : ids) {
        d += inst.dists[prev][id];
        prev = id;
    }
    d += inst.dists[prev][0];

    double diam = 0.0;
    for (int i : ids) {
        for (int j : ids) {
            diam = max(diam, inst.eucl[i][j]);
        }
    }

    return v.rental + d * v.fuel + (diam * 0.5) * v.radius;
}

static inline double route_cost(const Instance& inst, const Route& r) {
    return route_cost_ids_with_vehicle(inst, r.ids, r.v_idx);
}

static inline double solution_cost(const Instance& inst, const vector<Route>& sol) {
    double c = 0.0;
    for (const auto& r : sol) c += route_cost(inst, r);
    return c;
}

static inline bool choose_vehicle_for_route(const Instance& inst, Route& r, int topk = 5) {
    if (r.ids.empty()) return true;

    double req = route_weight(inst, r);
    vector<int> cand;
    candidate_vehicles(inst, req, cand, topk);

    for (int v_idx : cand) {
        if (is_feasible(inst, r.ids, v_idx)) {
            r.v_idx = v_idx;
            return true;
        }
    }

    for (int v_idx = 0; v_idx < (int)inst.vehicles.size(); ++v_idx) {
        if (inst.vehicles[v_idx].cap + 1e-6 < req) continue;
        if (is_feasible(inst, r.ids, v_idx)) {
            r.v_idx = v_idx;
            return true;
        }
    }
    return false;
}

static vector<Route> greedy_init(const Instance& inst) {
    vector<int> ids(inst.N);
    iota(ids.begin(), ids.end(), 1);

    sort(ids.begin(), ids.end(), [&](int a, int b) {
        const auto& A = inst.orders[a - 1];
        const auto& B = inst.orders[b - 1];
        if (A.start != B.start) return A.start < B.start;
        return A.end < B.end;
    });

    int fallback = (int)inst.vehicles.size() - 1;
    vector<Route> sol;
    sol.reserve(inst.N / 3 + 10);

    Route cur{fallback, {}};

    for (int id : ids) {
        cur.ids.push_back(id);

        vector<int> cand;
        candidate_vehicles(inst, route_weight(inst, cur), cand, 4);

        bool ok = false;
        for (int v_idx : cand) {
            if (is_feasible(inst, cur.ids, v_idx)) {
                cur.v_idx = v_idx;
                ok = true;
                break;
            }
        }
        if (ok) continue;

        cur.ids.pop_back();
        if (!cur.ids.empty()) sol.push_back(cur);

        cur = Route{fallback, {id}};
        choose_vehicle_for_route(inst, cur, 4);
    }

    if (!cur.ids.empty()) sol.push_back(cur);

    vector<Route> out;
    out.reserve(sol.size());
    for (auto& r : sol) {
        if (is_feasible(inst, r.ids, r.v_idx)) {
            out.push_back(r);
        } else {
            for (int id : r.ids) out.push_back(Route{fallback, {id}});
        }
    }
    return out;
}

static void repair_solution(const Instance& inst, vector<Route>& sol) {
    int fallback = (int)inst.vehicles.size() - 1;

    for (int rr = 0; rr < (int)sol.size(); ++rr) {
        int guard = 0;
        while (!sol[rr].ids.empty() && !is_feasible(inst, sol[rr].ids, sol[rr].v_idx) && guard++ < 140) {
            int worst_pos = 0;
            double worst_tight = -1.0;

            for (int p = 0; p < (int)sol[rr].ids.size(); ++p) {
                int id = sol[rr].ids[p];
                const auto& o = inst.orders[id - 1];
                double tight = 1.0 / max(1.0, (o.end - o.start));
                if (tight > worst_tight) {
                    worst_tight = tight;
                    worst_pos = p;
                }
            }

            int kicked = sol[rr].ids[worst_pos];
            sol[rr].ids.erase(sol[rr].ids.begin() + worst_pos);
            sol.push_back(Route{fallback, {kicked}});
        }

        if (sol[rr].ids.empty()) {
            sol.erase(sol.begin() + rr);
            --rr;
            continue;
        }

        choose_vehicle_for_route(inst, sol[rr], 5);
        if (!is_feasible(inst, sol[rr].ids, sol[rr].v_idx)) {
            vector<int> ids = sol[rr].ids;
            sol.erase(sol.begin() + rr);
            --rr;
            for (int id : ids) sol.push_back(Route{fallback, {id}});
        }
    }
}

static bool try_2opt(const Instance& inst, Route& r, mt19937& rng, int trials = 10) {
    int n = (int)r.ids.size();
    if (n < 4) return false;

    double bestC = route_cost(inst, r);
    bool improved = false;
    uniform_int_distribution<int> dist(0, n - 1);

    for (int t = 0; t < trials; ++t) {
        int i = dist(rng), j = dist(rng);
        if (i > j) swap(i, j);
        if (j - i < 2) continue;

        vector<int> tmp = r.ids;
        reverse(tmp.begin() + i, tmp.begin() + j + 1);

        if (!is_feasible(inst, tmp, r.v_idx)) continue;

        double c = route_cost_ids_with_vehicle(inst, tmp, r.v_idx);
        if (c + 1e-9 < bestC) {
            r.ids.swap(tmp);
            bestC = c;
            improved = true;
        }
    }
    return improved;
}

static bool try_cross_exchange_1(const Instance& inst, vector<Route>& sol, mt19937& rng, int trials = 35) {
    if (sol.size() < 2) return false;
    uniform_int_distribution<int> rDist(0, (int)sol.size() - 1);

    for (int t = 0; t < trials; ++t) {
        int a = rDist(rng), b = rDist(rng);
        if (a == b) continue;

        auto& A = sol[a];
        auto& B = sol[b];
        if (A.ids.empty() || B.ids.empty()) continue;

        int ia = uniform_int_distribution<int>(0, (int)A.ids.size() - 1)(rng);
        int ib = uniform_int_distribution<int>(0, (int)B.ids.size() - 1)(rng);

        int ca = A.ids[ia], cb = B.ids[ib];

        vector<int> nA = A.ids, nB = B.ids;
        nA[ia] = cb;
        nB[ib] = ca;

        Route rA{A.v_idx, nA};
        Route rB{B.v_idx, nB};

        if (!choose_vehicle_for_route(inst, rA, 6)) continue;
        if (!choose_vehicle_for_route(inst, rB, 6)) continue;
        if (!is_feasible(inst, rA.ids, rA.v_idx)) continue;
        if (!is_feasible(inst, rB.ids, rB.v_idx)) continue;

        double oldC = route_cost(inst, A) + route_cost(inst, B);
        double newC = route_cost(inst, rA) + route_cost(inst, rB);

        if (newC + 1e-9 < oldC) {
            A = std::move(rA);
            B = std::move(rB);
            return true;
        }
    }
    return false;
}

static bool try_2opt_star(const Instance& inst, vector<Route>& sol, mt19937& rng, int trials = 35) {
    if (sol.size() < 2) return false;
    uniform_int_distribution<int> rDist(0, (int)sol.size() - 1);

    for (int t = 0; t < trials; ++t) {
        int a = rDist(rng), b = rDist(rng);
        if (a == b) continue;

        auto& A = sol[a];
        auto& B = sol[b];
        if (A.ids.size() < 2 || B.ids.size() < 2) continue;

        int i = uniform_int_distribution<int>(0, (int)A.ids.size() - 2)(rng);
        int j = uniform_int_distribution<int>(0, (int)B.ids.size() - 2)(rng);

        vector<int> nA, nB;
        nA.insert(nA.end(), A.ids.begin(), A.ids.begin() + i + 1);
        nA.insert(nA.end(), B.ids.begin() + j + 1, B.ids.end());

        nB.insert(nB.end(), B.ids.begin(), B.ids.begin() + j + 1);
        nB.insert(nB.end(), A.ids.begin() + i + 1, A.ids.end());

        Route rA{A.v_idx, nA};
        Route rB{B.v_idx, nB};

        if (!choose_vehicle_for_route(inst, rA, 6)) continue;
        if (!choose_vehicle_for_route(inst, rB, 6)) continue;
        if (!is_feasible(inst, rA.ids, rA.v_idx)) continue;
        if (!is_feasible(inst, rB.ids, rB.v_idx)) continue;

        double oldC = route_cost(inst, A) + route_cost(inst, B);
        double newC = route_cost(inst, rA) + route_cost(inst, rB);

        if (newC + 1e-9 < oldC) {
            A = std::move(rA);
            B = std::move(rB);
            return true;
        }
    }
    return false;
}

struct InsBest {
    double best_score = 1e100;
    int best_pos = -1;
    int best_v = -1;
};

static InsBest best_insert_into_route(const Instance& inst, const Route& base, int id, mt19937& rng) {
    InsBest out;
    int m = (int)base.ids.size();

    vector<int> posCand(m + 1);
    iota(posCand.begin(), posCand.end(), 0);
    shuffle(posCand.begin(), posCand.end(), rng);
    if ((int)posCand.size() > 18) posCand.resize(18);

    double needW = route_weight(inst, base) + inst.orders[id - 1].weight;
    vector<int> candV;
    candidate_vehicles(inst, needW, candV, 5);

    for (int v_idx : candV) {
        for (int p : posCand) {
            vector<int> ids = base.ids;
            ids.insert(ids.begin() + p, id);

            double slack;
            if (!feasible_and_slack(inst, ids, v_idx, slack)) continue;

            double c = route_cost_ids_with_vehicle(inst, ids, v_idx);

            if (slack < 120.0) c += (120.0 - slack) * 8.0;
            if (slack < 60.0)  c += (60.0  - slack) * 16.0;

            if (c < out.best_score) {
                out.best_score = c;
                out.best_pos = p;
                out.best_v = v_idx;
            }
        }
    }

    return out;
}

static bool try_relocate_safe(const Instance& inst, vector<Route>& sol, mt19937& rng, int trials = 70) {
    if ((int)sol.size() < 2) return false;

    uniform_int_distribution<int> rDist(0, (int)sol.size() - 1);
    bool improved_any = false;

    for (int t = 0; t < trials; ++t) {
        int a = rDist(rng), b = rDist(rng);
        if (a == b) continue;
        if (sol[a].ids.empty()) continue;

        int pa = uniform_int_distribution<int>(0, (int)sol[a].ids.size() - 1)(rng);
        int id = sol[a].ids[pa];

        Route ra = sol[a];
        Route rb = sol[b];

        double oldAB = route_cost(inst, ra) + route_cost(inst, rb);

        ra.ids.erase(ra.ids.begin() + pa);
        if (!ra.ids.empty()) {
            if (!choose_vehicle_for_route(inst, ra, 6)) continue;
        } else {
            continue;
        }

        InsBest ins = best_insert_into_route(inst, rb, id, rng);
        if (ins.best_pos == -1) continue;

        rb.ids.insert(rb.ids.begin() + ins.best_pos, id);
        rb.v_idx = ins.best_v;
        if (!is_feasible(inst, rb.ids, rb.v_idx)) continue;

        double newAB = route_cost(inst, ra) + route_cost(inst, rb);

        if (newAB + 1e-9 < oldAB) {
            sol[a] = std::move(ra);
            sol[b] = std::move(rb);
            improved_any = true;
        }
    }

    return improved_any;
}

static void ruin_worst_light(const Instance& inst, vector<Route>& tmp, vector<int>& removed, mt19937& rng, int k_remove) {
    removed.clear();
    if (tmp.empty()) return;

    struct Cand {
        int id;
        double score;
    };

    vector<Cand> cand;
    cand.reserve(600);

    for (auto& r : tmp) {
        auto& ids = r.ids;
        int m = (int)ids.size();
        if (m < 2) continue;

        for (int p = 0; p < m; ++p) {
            int id = ids[p];
            int prev = (p == 0 ? 0 : ids[p - 1]);
            int nxt  = (p == m - 1 ? 0 : ids[p + 1]);
            double contrib = inst.dists[prev][id] + inst.dists[id][nxt] - inst.dists[prev][nxt];
            const auto& o = inst.orders[id - 1];
            double tight = 1.0 / max(80.0, (o.end - o.start));
            double s = contrib + 900.0 * tight;
            cand.push_back({id, s});
        }
    }

    if (cand.empty()) return;

    sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.score > b.score; });

    int take = min(k_remove, (int)cand.size());
    int band = min(35, (int)cand.size());

    vector<int> chosen;
    chosen.reserve(take);

    for (int t = 0; t < take; ++t) {
        int pick = uniform_int_distribution<int>(0, min(band - 1, (int)cand.size() - 1))(rng);
        int id = cand[pick].id;
        chosen.push_back(id);

        for (int rr = 0; rr < (int)tmp.size(); ++rr) {
            auto& ids = tmp[rr].ids;
            auto it = find(ids.begin(), ids.end(), id);
            if (it != ids.end()) {
                ids.erase(it);
                if (ids.empty()) tmp.erase(tmp.begin() + rr);
                break;
            }
        }

        cand.erase(remove_if(cand.begin(), cand.end(),
                             [&](const Cand& c) { return c.id == id; }),
                   cand.end());

        if (cand.empty()) break;
        sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.score > b.score; });
    }

    removed = std::move(chosen);
}

static void recreate_regret_lite(const Instance& inst, vector<Route>& tmp, vector<int>& removed, mt19937& rng) {
    int fallback = (int)inst.vehicles.size() - 1;

    while (!removed.empty()) {
        int scan = min(26, (int)removed.size());
        int bestPickIdx = 0;
        double bestReg = -1e100;

        int chosen_id = removed[0];
        int chosen_r = -1;
        InsBest chosen_ins;

        for (int i = 0; i < scan; ++i) {
            int id = removed[i];

            int best_r = -1;
            InsBest bestLocal;
            bestLocal.best_score = 1e100;
            double second = 1e100;

            if (!tmp.empty()) {
                vector<int> rlist(tmp.size());
                iota(rlist.begin(), rlist.end(), 0);
                shuffle(rlist.begin(), rlist.end(), rng);
                if ((int)rlist.size() > 30) rlist.resize(30);

                for (int rr : rlist) {
                    InsBest ins = best_insert_into_route(inst, tmp[rr], id, rng);
                    if (ins.best_pos == -1) continue;

                    if (ins.best_score < bestLocal.best_score) {
                        second = bestLocal.best_score;
                        bestLocal = ins;
                        best_r = rr;
                    } else if (ins.best_score < second) {
                        second = ins.best_score;
                    }
                }
            }

            double reg;
            if (best_r == -1) {
                reg = 1e50;
            } else {
                if (second > 9e99) second = bestLocal.best_score + 5000.0;
                reg = (second - bestLocal.best_score);
                const auto& o = inst.orders[id - 1];
                double tight = 1.0 / max(120.0, (o.end - o.start));
                reg += 250.0 * tight;
            }

            if (reg > bestReg) {
                bestReg = reg;
                bestPickIdx = i;
                chosen_id = id;
                chosen_r = best_r;
                chosen_ins = bestLocal;
            }
        }

        removed.erase(removed.begin() + bestPickIdx);

        if (chosen_r != -1) {
            tmp[chosen_r].ids.insert(tmp[chosen_r].ids.begin() + chosen_ins.best_pos, chosen_id);
            tmp[chosen_r].v_idx = chosen_ins.best_v;
        } else {
            tmp.push_back(Route{fallback, {chosen_id}});
        }
    }
}

static bool try_route_elimination(const Instance& inst, vector<Route>& sol, mt19937& rng, int maxRouteSize = 8) {
    if (sol.size() < 2) return false;

    vector<int> candidates;
    candidates.reserve(sol.size());

    for (int i = 0; i < (int)sol.size(); ++i) {
        int sz = (int)sol[i].ids.size();
        if (sz >= 1 && sz <= maxRouteSize) candidates.push_back(i);
    }

    if (candidates.empty()) return false;

    int ridx = candidates[uniform_int_distribution<int>(0, (int)candidates.size() - 1)(rng)];
    Route victim = sol[ridx];

    double oldTotal = solution_cost(inst, sol);

    vector<Route> tmp = sol;
    tmp.erase(tmp.begin() + ridx);

    vector<int> ins = victim.ids;
    shuffle(ins.begin(), ins.end(), rng);

    for (int id : ins) {
        int best_r = -1;
        InsBest bestIns;
        bestIns.best_score = 1e100;

        if (!tmp.empty()) {
            vector<int> rlist(tmp.size());
            iota(rlist.begin(), rlist.end(), 0);
            shuffle(rlist.begin(), rlist.end(), rng);
            if ((int)rlist.size() > 40) rlist.resize(40);

            for (int rr : rlist) {
                InsBest insb = best_insert_into_route(inst, tmp[rr], id, rng);
                if (insb.best_pos != -1 && insb.best_score < bestIns.best_score) {
                    bestIns = insb;
                    best_r = rr;
                }
            }
        }

        if (best_r != -1) {
            tmp[best_r].ids.insert(tmp[best_r].ids.begin() + bestIns.best_pos, id);
            tmp[best_r].v_idx = bestIns.best_v;
        } else {
            return false;
        }
    }

    for (auto& r : tmp) {
        if (!choose_vehicle_for_route(inst, r, 6)) return false;
        if (!is_feasible(inst, r.ids, r.v_idx)) return false;
    }

    double newTotal = solution_cost(inst, tmp);
    if (newTotal + 1e-9 < oldTotal) {
        sol = std::move(tmp);
        return true;
    }

    return false;
}

static void write_solution(const Instance& inst, const vector<Route>& sol, int idx, const string& out_dir) {
    ostringstream ss;
    ss << out_dir << "/solution_" << setfill('0') << setw(2) << idx << ".csv";
    ofstream f(ss.str());

    int mo = 0;
    for (const auto& r : sol) mo = max(mo, (int)r.ids.size());

    f << "family";
    for (int i = 1; i <= mo; ++i) f << ",order_" << i;
    f << "\n";

    for (const auto& r : sol) {
        f << inst.vehicles[r.v_idx].id;
        for (int id : r.ids) f << "," << id;
        for (int k = (int)r.ids.size(); k < mo; ++k) f << ",";
        f << "\n";
    }
}

static void solve_instance_time_limit(Instance& inst, int idx, const string& out_dir, int time_limit_sec, int seed) {
    mt19937 rng(seed + idx);
    vector<Route> best_sol = greedy_init(inst);
    repair_solution(inst, best_sol);
    double best_c = solution_cost(inst, best_sol);

    double T0 = 45.0;
    double Tend = 0.25;
    double alpha = 0.99965;

    double T = T0;
    vector<Route> cur_sol = best_sol;
    double cur_c = best_c;

    auto t_start = chrono::steady_clock::now();
    auto deadline = t_start + chrono::seconds(time_limit_sec);

    int it = 0;
    int last_checkpoint = 0;
    int fallback = (int)inst.vehicles.size() - 1;

    while (chrono::steady_clock::now() < deadline) {
        ++it;

        vector<Route> tmp = cur_sol;
        if (tmp.empty()) break;

        vector<int> removed;
        double uR = uniform_real_distribution<double>(0.0, 1.0)(rng);

        if (uR < 0.78) {
            int r_victim = uniform_int_distribution<int>(0, (int)tmp.size() - 1)(rng);
            for (int id : tmp[r_victim].ids) removed.push_back(id);
            tmp.erase(tmp.begin() + r_victim);

            int extra = min(14, inst.N / 35 + 2);
            for (int k = 0; k < extra && !tmp.empty(); ++k) {
                int rr = uniform_int_distribution<int>(0, (int)tmp.size() - 1)(rng);
                if (tmp[rr].ids.empty()) continue;
                int pp = uniform_int_distribution<int>(0, (int)tmp[rr].ids.size() - 1)(rng);
                removed.push_back(tmp[rr].ids[pp]);
                tmp[rr].ids.erase(tmp[rr].ids.begin() + pp);
                if (tmp[rr].ids.empty()) tmp.erase(tmp.begin() + rr);
            }
        } else {
            int k_remove = min(34, max(10, inst.N / 30));
            ruin_worst_light(inst, tmp, removed, rng, k_remove);
        }

        if (removed.empty()) continue;
        shuffle(removed.begin(), removed.end(), rng);

        if ((int)removed.size() <= 55) {
            recreate_regret_lite(inst, tmp, removed, rng);
        } else {
            for (int id : removed) {
                int best_r = -1;
                InsBest bestIns;
                bestIns.best_score = 1e100;

                if (!tmp.empty()) {
                    vector<int> rlist(tmp.size());
                    iota(rlist.begin(), rlist.end(), 0);
                    shuffle(rlist.begin(), rlist.end(), rng);
                    if ((int)rlist.size() > 32) rlist.resize(32);

                    for (int rr : rlist) {
                        InsBest insb = best_insert_into_route(inst, tmp[rr], id, rng);
                        if (insb.best_pos != -1 && insb.best_score < bestIns.best_score) {
                            bestIns = insb;
                            best_r = rr;
                        }
                    }
                }

                if (best_r != -1) {
                    tmp[best_r].ids.insert(tmp[best_r].ids.begin() + bestIns.best_pos, id);
                    tmp[best_r].v_idx = bestIns.best_v;
                } else {
                    tmp.push_back(Route{fallback, {id}});
                }
            }
        }

        repair_solution(inst, tmp);

        if (!tmp.empty()) {
            int n2opt = 115;
            for (int k = 0; k < n2opt; ++k) {
                int rr = uniform_int_distribution<int>(0, (int)tmp.size() - 1)(rng);
                try_2opt(inst, tmp[rr], rng, 7);
            }

            for (int k = 0; k < 18; ++k) {
                try_relocate_safe(inst, tmp, rng, 18);
                try_2opt_star(inst, tmp, rng, 18);
                try_cross_exchange_1(inst, tmp, rng, 18);
            }

            if ((it % 25) == 0) {
                for (int kk = 0; kk < 3; ++kk) {
                    if (try_route_elimination(inst, tmp, rng, 8)) {
                        for (int z = 0; z < 25 && !tmp.empty(); ++z) {
                            int rr = uniform_int_distribution<int>(0, (int)tmp.size() - 1)(rng);
                            try_2opt(inst, tmp[rr], rng, 6);
                        }
                    }
                }
            }
        }

        repair_solution(inst, tmp);
        double tmp_c = solution_cost(inst, tmp);

        bool accept = false;
        if (tmp_c < cur_c) {
            accept = true;
        } else {
            double delta = tmp_c - cur_c;
            double prob = exp(-delta / max(1e-9, T));
            double u = uniform_real_distribution<double>(0.0, 1.0)(rng);
            if (u < prob) accept = true;
        }

        if (accept) {
            cur_sol = std::move(tmp);
            cur_c = tmp_c;
        }

        if (cur_c < best_c) {
            best_c = cur_c;
            best_sol = cur_sol;
        }

        T = max(Tend, T * alpha);

        if (it - last_checkpoint >= 1800) {
            write_solution(inst, best_sol, idx, out_dir);
            last_checkpoint = it;
        }
    }

    write_solution(inst, best_sol, idx, out_dir);
}

static vector<Vehicle> read_vehicles(const string& veh_path) {
    ifstream vf(veh_path);
    if (!vf) {
        cerr << "Error reading vehicles file: " << veh_path << "\n";
        exit(1);
    }

    string line;
    getline(vf, line);

    vector<Vehicle> vehicles;
    while (getline(vf, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string t;
        vector<string> v;
        while (getline(ss, t, ',')) v.push_back(t);
        if (v.size() < 15) continue;

        Vehicle V;
        V.id = stoi(v[0]);
        V.cap = stoi(v[1]);
        V.rental = stod(v[2]);
        V.fuel = stod(v[3]);
        V.radius = stod(v[4]);
        V.speed = stod(v[5]);

        V.cosc[0] = stod(v[7]);
        V.sinc[0] = stod(v[8]);
        V.cosc[1] = stod(v[9]);
        V.sinc[1] = stod(v[10]);
        V.cosc[2] = stod(v[11]);
        V.sinc[2] = stod(v[12]);
        V.cosc[3] = stod(v[13]);
        V.sinc[3] = stod(v[14]);

        vehicles.push_back(V);
    }

    return vehicles;
}

static Instance read_instance(const string& inst_path, const vector<Vehicle>& vehicles) {
    Instance inst;
    inst.vehicles = vehicles;

    ifstream inf(inst_path);
    if (!inf) {
        cerr << "Error reading instance file: " << inst_path << "\n";
        exit(1);
    }

    string line;
    getline(inf, line);

    while (getline(inf, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string t;
        vector<string> v;
        while (getline(ss, t, ',')) v.push_back(t);
        if (v.size() < 7) continue;

        int id = stoi(v[0]);
        double lat = stod(v[1]);
        double lon = stod(v[2]);

        Order o;
        o.id = id;
        o.y = RHO * (lat - DEPOT_LAT) * PI / 180.0;
        o.x = RHO * cos(DEPOT_LAT * PI / 180.0) * (lon - DEPOT_LON) * PI / 180.0;

        if (id == 0) {
            inst.depot = o;
        } else {
            o.weight = stod(v[3]);
            o.start = stod(v[4]);
            o.end = stod(v[5]);
            o.dur = stod(v[6]);
            inst.orders.push_back(o);
        }
    }

    inst.N = (int)inst.orders.size();

    for (int a = 0; a <= inst.N; ++a) {
        for (int b = 0; b <= inst.N; ++b) {
            const Order& p1 = (a == 0 ? inst.depot : inst.orders[a - 1]);
            const Order& p2 = (b == 0 ? inst.depot : inst.orders[b - 1]);
            inst.dists[a][b] = fabs(p1.x - p2.x) + fabs(p1.y - p2.y);
            double dx = p1.x - p2.x, dy = p1.y - p2.y;
            inst.eucl[a][b] = sqrt(dx * dx + dy * dy);
        }
    }

    return inst;
}

static void print_usage(const char* prog) {
    cerr << "Usage: " << prog
         << " [--instances_dir PATH]"
         << " [--vehicles PATH]"
         << " [--out_dir PATH]"
         << " [--time_limit SECONDS]"
         << " [--seed INT]"
         << " [--first_instance INT]"
         << " [--last_instance INT]"
         << "\n";
}

static Config parse_args(int argc, char** argv) {
    Config cfg;

    for (int i = 1; i < argc; ++i) {
        string a = argv[i];

        auto need_value = [&](const string& opt) -> string {
            if (i + 1 >= argc) {
                cerr << "Missing value for " << opt << "\n";
                print_usage(argv[0]);
                exit(1);
            }
            return argv[++i];
        };

        if (a == "--instances_dir") cfg.instances_dir = need_value(a);
        else if (a == "--vehicles") cfg.vehicles_file = need_value(a);
        else if (a == "--out_dir") cfg.output_dir = need_value(a);
        else if (a == "--time_limit") cfg.time_limit_sec = stoi(need_value(a));
        else if (a == "--seed") cfg.seed = stoi(need_value(a));
        else if (a == "--first_instance") cfg.first_instance = stoi(need_value(a));
        else if (a == "--last_instance") cfg.last_instance = stoi(need_value(a));
        else if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            exit(0);
        } else {
            cerr << "Unknown option: " << a << "\n";
            print_usage(argv[0]);
            exit(1);
        }
    }

    return cfg;
}

int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);

    std::filesystem::create_directories(cfg.output_dir);
    auto vehicles = read_vehicles(cfg.vehicles_file);

    for (int i = cfg.first_instance; i <= cfg.last_instance; ++i) {
        ostringstream ss;
        ss << cfg.instances_dir << "/instance_" << setfill('0') << setw(2) << i << ".csv";

        cerr << "Solving instance " << i << "...\n";
        Instance inst = read_instance(ss.str(), vehicles);
        solve_instance_time_limit(inst, i, cfg.output_dir, cfg.time_limit_sec, cfg.seed);
    }

    return 0;
}
