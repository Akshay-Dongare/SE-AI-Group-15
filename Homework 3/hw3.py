#!/usr/bin/env python3 -B
"""hw3.py: Welford Num vs reservoir Num"""
import random, math, glob, statistics
from ez import csv, Data, shuffle, Num, col
from sa import sa
from locals import ls, lsRminus, saRplus
from stats import top
import ez

class WelfordNum(Num):
  """Welford online mean/variance. No reservoir.
  Approximate lo/hi as mu +/- 3*sd."""

  def __init__(self, mx=None):
    list.__init__(self)       # empty list (unused)
    self.n    = 0
    self.mu   = 0.0
    self.m2   = 0.0
    self.seen = 0
    self.mx   = mx or 256

  def add(self, v):
    if v == "?": return v
    self.seen += 1
    self.n   += 1
    # TODO: Welford incremental update
    #   delta   = v - self.mu
    #   self.mu += delta / self.n
    #   delta2  = v - self.mu
    #   self.m2 += delta * delta2
    return v

  def sub(self, v):
    if v == "?": return v
    self.seen -= 1
    if self.n <= 1:
      self.n, self.mu, self.m2 = 0, 0.0, 0.0
      return v
    # TODO: reverse Welford
    #   delta   = v - self.mu
    #   self.n -= 1
    #   self.mu -= delta / self.n
    #   delta2  = v - self.mu
    #   self.m2 -= delta * delta2
    return v

  def mid(self):
    return 0  # TODO: return self.mu

  def spread(self):
    return 0  # TODO: sqrt(m2/(n-1)) if n>1 else 0

  def _lo(self): return self.mu - 3*self.spread()
  def _hi(self): return self.mu + 3*self.spread()

  def norm(self, v):
    if v == "?": return v
    lo, hi = self._lo(), self._hi()
    # TODO: normalize v into 0..1 using lo, hi
    #   return 0 if lo==hi else max(0,min(1,...))
    return v

  def pick(self, v=None):
    # TODO: gaussian perturbation around v or mu
    #   base = self.mu if v is None or v=="?" else v
    #   result = random.gauss(base, self.spread())
    #   clamp to [_lo(), _hi()]
    return self.mu

  def distx(self, u, v):
    if u == v == "?": return 1
    u, v = self.norm(u), self.norm(v)
    u = u if u != "?" else (0 if v > 0.5 else 1)
    v = v if v != "?" else (0 if u > 0.5 else 1)
    return abs(u - v)

  def like(self, v, prior=0):
    s = self.spread() + 1e-32
    return ((1/math.sqrt(2*math.pi*s*s))
            * math.exp(-((v-self.mu)**2)/(2*s*s)))

# --- monkey-patch ---
_original_col = ez.col
def welford_col(s):
  if s[0].isupper(): return WelfordNum()
  return _original_col(s)

def run_tour(files, use_welford=False):
  ez.col = welford_col if use_welford else _original_col
  ALGOS   = [sa, ls, lsRminus, saRplus]
  REPEATS = 20; SAMPLE = 50
  wins = {a.__name__: 0 for a in ALGOS}

  for f in sorted(files):
    try:
      d0 = Data(csv(f))
      if len(d0.rows) < SAMPLE: continue
    except Exception:
      continue

    baseline = []
    for _ in range(REPEATS):
      rows = shuffle(d0.rows[:])[:SAMPLE]
      d1   = Data([d0.cols.names] + rows)
      for r in d1.rows:
        baseline.append(d1.disty(r))
    sd = statistics.stdev(baseline) \
         if len(baseline) > 1 else 1

    seen = {a.__name__: [] for a in ALGOS}
    for _ in range(REPEATS):
      rows = shuffle(d0.rows[:])[:SAMPLE]
      d1 = Data([d0.cols.names] + rows)
      for algo in ALGOS:
        # TODO: run algo, collect final energy
        #   for h, e, row in algo(d1): pass
        #   seen[algo.__name__].append(int(100*e))
        pass

    # TODO: winners = top(seen, eps=0.35 * sd)
    # TODO: for w in winners: wins[w] += 1

  for name in sorted(wins):
    print(f"  {name:>12} {wins[name]:>6}")

def eg__compare(d:filename):
  "parametric vs non-parametric tournament"
  files = glob.glob(d + "/*/*.csv")
  print(f"found {len(files)} csv files")
  print("\n=== RESERVOIR (non-parametric) ===")
  run_tour(files, use_welford=False)
  print("\n=== WELFORD (Gaussian) ===")
  run_tour(files, use_welford=True)

if __name__ == "__main__":
  from ez import main
  main(globals())