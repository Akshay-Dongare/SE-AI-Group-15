#!/usr/bin/env python3 -B
"""hw1a.py: accuracy under class imbalance"""
import random
from stats import Confuse, confuse, confused

random.seed(1)

RATIOS = [              # (num_pos, num_neg)
  (50,  50),
  (10,  90),
  (5,   95),
  (1,   99),
  (1,   999)]

TP_RATE = 0.70          # classifier catches 70% of +
FP_RATE = 0.05          # classifier false-alarms 5%

print(f"{'ratio':>10} {'acc':>5} {'pd':>5}"
      f" {'pf':>5} {'prec':>5}")
print("-" * 40)

for n_pos, n_neg in RATIOS:
  cf = Confuse()
  for _ in range(n_pos):
    # TODO: predict "pos" with prob TP_RATE, else "neg"
    got = "???"
    confuse(cf, "pos", got)
  for _ in range(n_neg):
    # TODO: predict "pos" with prob FP_RATE, else "neg"
    got = "???"
    confuse(cf, "neg", got)
  summary = confused(cf, summary=True)
  # TODO: print ratio, summary.acc, summary.pd,
  #       summary.pf, summary.prec