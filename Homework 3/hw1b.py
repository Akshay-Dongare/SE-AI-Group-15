#!/usr/bin/env python3 -B
"""hw1b.py: top() with pragmatic eps"""
import random, math, statistics
from stats import top

random.seed(1)

def weibull(n=20):
  shape = random.uniform(0.5, 3)
  scale = random.uniform(1, 4)
  return [min(10,
    scale*(-math.log(random.random()))**(1/shape)*2.5)
    for _ in range(n)]

sizes = []
for trial in range(50):
  rxs = {i: weibull() for i in range(20)}
  pooled = [x for xs in rxs.values() for x in xs]
  sd     = statistics.stdev(pooled)

  eps = 0.35 * sd
  winners = top(rxs, eps=eps)
  sizes.append(len(winners))

  winner_means = [statistics.mean(rxs[w]) for w in winners]
  if winner_means:
    center = statistics.mean(winner_means)
    assert all(abs(m - center) <= eps for m in winner_means)

print(f"avg winners: {sum(sizes)/len(sizes):.1f}/20")
print(f"min winners: {min(sizes)}")
print(f"max winners: {max(sizes)}")
print("larger eps -> more winners (harder to justify practical differences)")