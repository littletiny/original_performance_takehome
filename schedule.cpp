// Offline serial schedule generation with fixed multi-cycle dispatch regions.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <queue>
#include <random>
#include <utility>
#include <vector>

extern "C" int schedule_search(
    int n, int ne, const int64_t* sources, const int64_t* dests,
    const int64_t* lags, const int64_t* durations, const int64_t* usage,
    const double* initial_keys, int iterations, uint64_t seed,
    double noise, int64_t* best_out, int64_t* current_out) {
  constexpr int horizon = 16000;
  constexpr int caps[5] = {12, 6, 2, 2, 1};
  std::vector<std::vector<std::pair<int,int>>> successors[2];
  std::vector<int> indegrees[2];
  std::vector<std::array<int,3>> resources[2];
  for (int dir = 0; dir < 2; ++dir) {
    successors[dir].resize(n);
    indegrees[dir].assign(n, 0);
    resources[dir].resize(n * 33 * 5);
  }
  for (int e = 0; e < ne; ++e) {
    int a = sources[e], b = dests[e], lag = lags[e];
    successors[0][a].emplace_back(b, lag);
    successors[1][b].emplace_back(a, lag + durations[b] - durations[a]);
    indegrees[0][b]++;
    indegrees[1][a]++;
  }
  std::vector<std::vector<std::array<int,3>>> demands[2];
  for (int dir = 0; dir < 2; ++dir) {
    demands[dir].resize(n);
    for (int i = 0; i < n; ++i)
      for (int off = 0; off <= durations[i]; ++off)
        for (int e = 0; e < 5; ++e) {
          int amount = usage[(i*33 + off)*5 + e];
          if (amount) demands[dir][i].push_back({dir ? int(durations[i])-off : off, e, amount});
        }
  }
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> jitter(-noise, noise);
  std::vector<int> best(n), current(n), placed(n), releases(n), pending;
  std::vector<double> keys(initial_keys, initial_keys+n);
  int best_end = horizon;
  int current_end = horizon;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    int dir = iteration % 2;
    if (iteration) {
      for (int i = 0; i < n; ++i) {
        keys[i] = dir ? current_end - 1 - current[i] - durations[i] : current[i];
        keys[i] += jitter(rng);
      }
    }
    pending = indegrees[dir];
    std::fill(releases.begin(), releases.end(), 0);
    std::vector<std::array<int,5>> occupied(horizon);
    using Item = std::pair<double,int>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
    for (int i = 0; i < n; ++i) if (!pending[i]) heap.emplace(keys[i], i);
    int issued = 0, end = 0;
    while (!heap.empty()) {
      int i = heap.top().second;
      heap.pop();
      int cycle = releases[i];
      while (true) {
        if (cycle + durations[i] >= horizon) return -2;
        bool good = true;
        for (auto row : demands[dir][i]) {
          if (occupied[cycle + row[0]][row[1]] + row[2] > caps[row[1]]) {
            good = false;
            break;
          }
        }
        if (good) break;
        ++cycle;
      }
      placed[i] = cycle;
      end = std::max(end, cycle + int(durations[i]) + 1);
      for (auto row : demands[dir][i]) occupied[cycle+row[0]][row[1]] += row[2];
      for (auto [child, lag] : successors[dir][i]) {
        releases[child] = std::max(releases[child], cycle + lag);
        if (!--pending[child]) heap.emplace(keys[child], child);
      }
      ++issued;
    }
    if (issued != n) return -1;
    current_end = end;
    for (int i = 0; i < n; ++i) current[i] = dir ? end - 1 - placed[i] - durations[i] : placed[i];
    if (end < best_end) { best_end = end; best = current; }
  }
  for (int i = 0; i < n; ++i) { best_out[i] = best[i]; current_out[i] = current[i]; }
  return best_end;
}
