#!/usr/bin/env python3 -B
# Usage: python3 -B my_hc.py 1 ~/gits/moot/optimize/misc/auto93.csv

import random, sys, xai

# --- Setup ---
xai.the.data = sys.argv[2]
random.seed(int(sys.argv[1]))
data = xai.Data(xai.csv(xai.the.data))

# Helper to get rounded distance to heaven (Minimize Y)
def Y(r): return round(xai.disty(data, r), 2)

# --- Exercise 2: Reporting Progress ---
def report(what, rows):
    a = sorted(rows[:], key=Y)
    print(f":n {len(a):4} :lo {Y(a[0]):5.2f} :mid {Y(a[len(a)//2]):5.2f}", what)

# --- Exercise 3: Finding Extremes (Good vs Bad) ---
def extremes(rows):
    rows.sort(key=Y)
    n = len(rows) // 10
    return rows[n], rows[9*n]  # Return 10th percentile (good) and 90th (bad)

# --- Exercise 4: Projection Geometry ---
def project(r, ok, no):
    a = xai.distx(data, r, ok)   # Distance to "good"
    b = xai.distx(data, r, no)   # Distance to "bad"
    c = xai.distx(data, ok, no)  # Distance between good and bad
    
    # Law of Cosines to project r onto the line between ok and no
    # (a^2 + c^2 - b^2) / (2c)
    return (a**2 + c**2 - b**2) / (2*c + 1e-32)

# --- Exercise 5: Pruning the Population ---
def prune(rows, ok, no):
    # Sort rows by their projection (smaller value = closer to "ok")
    rows.sort(key=lambda r: project(r, ok, no))
    n = len(rows)
    return rows[:n//2]  # Keep the better half

# --- Exercise 6: Convergence Detection Helpers ---
def top(a): a.sort(); return a[0]
def mid(a): a.sort(); n = len(a)//10; return a[5*n]
def sd(a):  a.sort(); n = len(a)//10; return (a[9*n] - a[n]) / 2.56

# --- Run Baselines (Exercise 1) ---
print("\n--- Baselines ---")
report("baseline", data.rows)
report("sample", xai.shuffle(data.rows)[:30])

# --- Calculate Convergence Threshold (Exercise 6) ---
cohen = 0.35
eps = sd([Y(r) for r in data.rows]) * cohen
print(f"\nConvergence Threshold (eps): {eps:.2f}")

# --- Exercise 9: Multiple Runs (Statistics) ---
print(f"\n{'score':5} {'evals'}  {'Run Status'}")
print("-" * 30)

for run in range(20):
    # --- Exercise 7: Single Hill-Climbing Setup ---
    rows = xai.shuffle(data.rows[:])
    labelled = []
    budget = 100
    step = 5
    b4 = 1e32 # Init "before" score to huge number
    
    # Start the Hill Climber
    while len(labelled) < budget:
        # 1. Label new random points (Step)
        new_points = rows[:step]
        rows = rows[step:] # Remove them from the pool
        labelled += new_points
        
        # 2. Find direction (Extremes)
        ok, no = extremes(labelled)
        
        # 3. Prune the remaining pool
        rows = prune(rows, ok, no)
        
        # --- Exercise 8: Convergence Check ---
        now = top([Y(r) for r in labelled]) # Current best score
        
        # Stop if improvement is negligible (less than eps)
        if abs(b4 - now) < eps:
            break 
            
        b4 = now
    
    # Report result for this run
    print(f"{top([Y(r) for r in labelled]):5.2f} {len(labelled):5}  Run {run+1}")