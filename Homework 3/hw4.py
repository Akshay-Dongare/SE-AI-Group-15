#!/usr/bin/env python3 -B
"""hw4.py: hyperparameter sensitivity (grad only)"""
import random, glob, statistics
from ez import csv, Data, shuffle, main, filename
from sa import sa, oneplus1
from locals import ls
from stats import top

SA_MS    = [0.1, 0.3, 0.5, 0.7, 0.9]
LS_RS    = [0, 25, 50, 100, 200]
REPEATS  = 20
SAMPLE   = 50

def eg__hparam(d:filename):
  "hyperparameter sensitivity across MOOT"
  files = glob.glob(d + "/*/*.csv")
  print(f"found {len(files)} csv files")

  # treatment names are tuples
  treatments = ([("sa",m)  for m in SA_MS] +
                [("ls",rs) for rs in LS_RS])
  wins = {t: 0 for t in treatments}

  for f in sorted(files):
    try:
      d0 = Data(csv(f))
      if len(d0.rows) < SAMPLE: continue
    except Exception:
      continue

    # --- baseline eps ---
    baseline = []
    for _ in range(REPEATS):
      rows = shuffle(d0.rows[:])[:SAMPLE]
      d1   = Data([d0.cols.names] + rows)
      for r in d1.rows:
        baseline.append(d1.disty(r))
    sd = statistics.stdev(baseline) \
         if len(baseline) > 1 else 1

    seen = {t: [] for t in treatments}
    for _ in range(REPEATS):
      rows = shuffle(d0.rows[:])[:SAMPLE]
      d1   = Data([d0.cols.names] + rows)
      for name, param in treatments:
        # TODO: if name=="sa":
        #   run sa(d1, m=param), get final e
        # TODO: if name=="ls":
        #   run ls(d1, restarts=param), get final e
        # seen[(name,param)].append(int(100*e))
        pass

    # TODO: winners = top(seen, eps=0.35 * sd)
    # TODO: for w in winners: wins[w] += 1

  print(f"\n{'treatment':>15} {'wins':>6}")
  print("-" * 25)
  for t in sorted(wins, key=lambda t: -wins[t]):
    print(f"{str(t):>15} {wins[t]:>6}")

if __name__ == "__main__": main(globals())