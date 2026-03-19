#!/usr/bin/env python3 -B
"""hw5.py: sample size sensitivity (grad only)"""
import random, glob, statistics
from ez import csv, Data, shuffle, main, filename
from sa import sa
from locals import ls, lsRminus, saRplus
from stats import top

ALGOS   = [sa, ls, lsRminus, saRplus]
SAMPLES = [30, 50, 100, 200]
REPEATS = 20

def eg__sample(d:filename):
  "sample size sensitivity across MOOT"
  files = glob.glob(d + "/*/*.csv")
  print(f"found {len(files)} csv files")

  # treatments: (algo_name, sample_size)
  treatments = [(a.__name__, n)
                for a in ALGOS for n in SAMPLES]
  wins = {t: 0 for t in treatments}
  algo_by_name = {a.__name__: a for a in ALGOS}

  for f in sorted(files):
    try:
      d0 = Data(csv(f))
      if len(d0.rows) < max(SAMPLES): continue
    except Exception:
      continue

    # --- baseline eps from largest sample ---
    baseline = []
    for _ in range(REPEATS):
      rows = shuffle(d0.rows[:])[:max(SAMPLES)]
      d1   = Data([d0.cols.names] + rows)
      for r in d1.rows:
        baseline.append(d1.disty(r))
    sd = statistics.stdev(baseline) \
         if len(baseline) > 1 else 1

    seen = {t: [] for t in treatments}
    for _ in range(REPEATS):
      for name, n in treatments:
        rows = shuffle(d0.rows[:])[:n]
        d1   = Data([d0.cols.names] + rows)
        algo = algo_by_name[name]
        # TODO: run algo(d1) to get final energy
        #   for h, e, row in algo(d1): pass
        #   seen[(name, n)].append(int(100*e))
        pass

    # TODO: winners = top(seen, eps=0.35 * sd)
    #
    # tiebreak: prefer smallest sample
    # TODO:
    #   best_n  = min(n for (_,n) in winners)
    #   winners = {w for w in winners if w[1]==best_n}
    #
    # TODO: for w in winners: wins[w] += 1

  print(f"\n{'treatment':>20} {'wins':>6}")
  print("-" * 30)
  for t in sorted(wins, key=lambda t: -wins[t]):
    print(f"{str(t):>20} {wins[t]:>6}")

if __name__ == "__main__": main(globals())