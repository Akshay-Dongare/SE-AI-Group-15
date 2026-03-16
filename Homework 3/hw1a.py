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
    got = "pos" if random.random() < TP_RATE else "neg"
    confuse(cf, "pos", got)
  for _ in range(n_neg):
    got = "pos" if random.random() < FP_RATE else "neg"
    confuse(cf, "neg", got)
  summary = confused(cf, summary=True)

  ratio = f"{n_pos}/{n_neg}"
  print(f"{ratio:>10} {summary.acc:5d} {summary.pd:5d}"
        f" {summary.pf:5d} {summary.prec:5d}")