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

  # baseline = pool all observations
  pooled = []  # TODO: flatten all rxs values
  sd     = 0   # TODO: statistics.stdev(pooled)

  winners = top(rxs, eps=0.35 * sd)
  sizes.append(len(winners))

  # TODO: check that all winner means are
  #       within eps of each other

print(f"avg winners: {sum(sizes)/len(sizes):.1f}/20")
print(f"min winners: {min(sizes)}")
print(f"max winners: {max(sizes)}")
# TODO: does larger eps -> more or fewer winners? why?