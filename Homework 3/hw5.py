#!/usr/bin/env python3 -B
"""hw5.py: sample size sensitivity (grad only)"""
import random, glob, statistics
from ez import csv, Data, shuffle, main
from sa import sa
from locals import ls, lsRminus, saRplus
from stats import top

random.seed(1)

ALGOS   = [sa, ls, lsRminus, saRplus]
SAMPLES = [30, 50, 100, 200]
REPEATS = 20

def eg__sample(d:str):
  "sample size sensitivity across MOOT"
  files = glob.glob(d + "/*/*.csv")
  print(f"found {len(files)} csv files")

  # treatments: (algo_name, sample_size)
  treatments = [(a.__name__, n)
                for a in ALGOS for n in SAMPLES]
  wins = {t: 0 for t in treatments}
  algo_by_name = {a.__name__: a for a in ALGOS}

  for i, f in enumerate(sorted(files), 1):
    print(f"[{i}/{len(files)}] Processing {f} ...", flush=True)
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
        e = None
        for _, e, _ in algo(d1):
          pass
        if e is not None:
          seen[(name, n)].append(int(100*e))

    winners = top(seen, eps=0.35 * sd)
    best_n  = min(n for (_, n) in winners)
    winners = {w for w in winners if w[1] == best_n}
    for w in winners:
      wins[w] += 1

  # --- PRINT AND SAVE RESULTS ---
  result_text = f"\n{'treatment':>20} {'wins':>6}\n" + "-" * 30 + "\n"
  for t in sorted(wins, key=lambda t: -wins[t]):
    result_text += f"{str(t):>20} {wins[t]:>6}\n"

  print(result_text)

  with open("hw5_final_results.txt", "w") as f:
    f.write(result_text)
  print("\n*** Results safely written to hw5_final_results.txt ***")

if __name__ == "__main__": main(globals())