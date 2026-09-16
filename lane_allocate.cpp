// Offline scratch allocation with a separate lifetime for every vector lane.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

extern "C" int allocate_lanes_native(
    int n, int horizon, int capacity, const int64_t* sizes,
    const int64_t* begins, const int64_t* ends, int policy, int64_t* bases) {
  const int stride = (horizon + 63) / 64;
  using Part = std::pair<int, uint64_t>;
  std::vector<std::vector<Part>> masks(n * 8);
  std::vector<int> start(n, horizon), finish(n, 0), order;
  std::vector<uint64_t> occupied(capacity * stride, 0);
  for (int v = 0; v < n; ++v) {
    bases[v] = -1;
    for (int lane = 0; lane < sizes[v]; ++lane) {
      const int a = begins[v*8+lane], b = ends[v*8+lane];
      if (a >= b) continue;
      start[v] = std::min(start[v], a);
      finish[v] = std::max(finish[v], b);
      for (int word = a / 64; word <= (b-1) / 64; ++word) {
        const int lo = std::max(a, word*64) - word*64;
        const int hi = std::min(b, (word+1)*64) - word*64;
        const uint64_t upper = hi == 64 ? ~uint64_t(0) : (uint64_t(1) << hi)-1;
        masks[v*8+lane].emplace_back(word, upper & (~uint64_t(0) << lo));
      }
    }
    if (finish[v]) order.push_back(v);
  }
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    auto key = [&](int v) {
      if (policy >= 80) {
        if (policy == 81) return std::make_tuple(-int(sizes[v])*(finish[v]-start[v]), -int(sizes[v]), start[v], v);
        if (policy == 82) return std::make_tuple(-finish[v], -int(sizes[v]), start[v], v);
        if (policy == 83) return std::make_tuple(start[v], -int(sizes[v]), -finish[v], v);
        return std::make_tuple(start[v]-finish[v], -int(sizes[v]), start[v], v);
      }
      if (policy >= 8) {
        uint64_t x = uint64_t(v) + uint64_t(policy)*0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30))*0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27))*0x94d049bb133111ebULL;
        x ^= x >> 31;
        const int spread = 4 << ((policy-8) % 4);
        const int jitter = int(x % (2*spread+1)) - spread;
        return std::make_tuple(start[v]-finish[v]+jitter, -int(sizes[v]), start[v], v);
      }
      if (policy == 2) return std::make_tuple(-finish[v], -int(sizes[v]), start[v], v);
      if (policy == 3) return std::make_tuple(start[v]-finish[v], -int(sizes[v]), start[v], v);
      if (policy == 4) return std::make_tuple(int(sizes[v]), start[v], -finish[v], v);
      if (policy == 5) return std::make_tuple(-int(sizes[v]), start[v], -finish[v], v);
      return std::make_tuple(start[v], -int(sizes[v]), policy == 1 ? -finish[v] : finish[v], v);
    };
    return key(a) < key(b);
  });
  for (int v : order) {
    auto fits = [&](int base) {
      for (int lane = 0; lane < sizes[v]; ++lane)
        for (auto [word, mask] : masks[v*8+lane])
          if (occupied[(base+lane)*stride+word] & mask) return false;
      return true;
    };
    int chosen = -1;
    if (policy >= 80) {
      int best_gap = std::numeric_limits<int>::max();
      for (int base = 0; base+sizes[v] <= capacity; ++base) {
        if (!fits(base)) continue;
        int gap = 0;
        for (int lane = 0; lane < sizes[v]; ++lane) {
          if (masks[v*8+lane].empty()) continue;
          const int a = begins[v*8+lane], b = ends[v*8+lane];
          const auto* cells = &occupied[(base+lane)*stride];
          int left = 0, right = horizon;
          if (a) {
            int word = (a-1)/64, bit = (a-1)%64;
            uint64_t bits = cells[word] & (bit == 63 ? ~uint64_t(0) : (uint64_t(1) << (bit+1))-1);
            while (!bits && word > 0) bits = cells[--word];
            if (bits) left = word*64+64-__builtin_clzll(bits);
          }
          if (b < horizon) {
            int word = b/64;
            uint64_t bits = cells[word] & (~uint64_t(0) << (b%64));
            while (!bits && word+1 < stride) bits = cells[++word];
            if (bits) right = word*64+__builtin_ctzll(bits);
          }
          gap += a-left+right-b;
        }
        if (gap < best_gap || (gap == best_gap && sizes[v] == 1)) {
          best_gap = gap; chosen = base;
        }
      }
    } else if (sizes[v] == 1 && policy != 7 && (policy < 8 || policy % 4 != 3)) {
      for (int base = capacity-1; base >= 0; --base)
        if (fits(base)) { chosen = base; break; }
    } else {
      if ((policy == 6 || policy >= 8 && policy % 3 == 0) && sizes[v] == 8)
        for (int base = policy >= 8 ? (policy/3) % 8 : 0; base+sizes[v] <= capacity; base += 8)
          if (fits(base)) { chosen = base; break; }
      if (chosen < 0)
        for (int base = 0; base+sizes[v] <= capacity; ++base)
          if (fits(base)) { chosen = base; break; }
    }
    if (chosen < 0) return v+1;
    bases[v] = chosen;
    for (int lane = 0; lane < sizes[v]; ++lane)
      for (auto [word, mask] : masks[v*8+lane])
        occupied[(chosen+lane)*stride+word] |= mask;
  }
  return 0;
}
