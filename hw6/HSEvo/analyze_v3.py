import json
import numpy as np
from scipy import stats

with open('experiment_results_v3.json', 'r') as f:
    data = json.load(f)

datasets = list(data.keys())
algs = ["random", "greedy", "motpe", "nsgaii", "hsevo"]
seeds = ["42", "123", "7"]

print("=" * 80)
print("COMPREHENSIVE ANALYSIS OF EXPERIMENT RESULTS v3")
print("=" * 80)

# 1. Mean HV per algorithm per dataset (averaged across seeds)
print("\n--- TABLE: Mean HV (across 3 seeds) ---")
print(f"{'Dataset':<10}", end="")
for a in algs:
    print(f"  {a:>8}", end="")
print()

all_final_hvs = {a: [] for a in algs}
for ds in datasets:
    print(f"{ds:<10}", end="")
    for a in algs:
        vals = []
        for s in seeds:
            ts = data[ds][a][s]["hv_timeseries"]
            vals.append(ts[-1])
        mean_val = np.mean(vals)
        all_final_hvs[a].append(mean_val)
        print(f"  {mean_val:>8.3f}", end="")
    print()

print(f"\n{'Average':<10}", end="")
for a in algs:
    print(f"  {np.mean(all_final_hvs[a]):>8.3f}", end="")
print()

# 2. Wall-clock time
print("\n\n--- TABLE: Mean Wall-Clock Time (seconds) ---")
print(f"{'Dataset':<10}", end="")
for a in algs:
    print(f"  {a:>8}", end="")
print()

all_times = {a: [] for a in algs}
for ds in datasets:
    print(f"{ds:<10}", end="")
    for a in algs:
        vals = []
        for s in seeds:
            vals.append(data[ds][a][s]["wall_clock_s"])
        mean_val = np.mean(vals)
        all_times[a].append(mean_val)
        print(f"  {mean_val:>8.2f}", end="")
    print()

print(f"\n{'Total':<10}", end="")
for a in algs:
    print(f"  {np.sum(all_times[a]):>8.2f}", end="")
print()

# 3. Wilcoxon signed-rank test: HSEvo vs each baseline
print("\n\n--- WILCOXON SIGNED-RANK TEST (HSEvo vs. each baseline) ---")
print("  (paired by dataset, using mean HV across seeds)")
hsevo_vals = np.array(all_final_hvs["hsevo"])
for baseline in ["random", "greedy", "motpe", "nsgaii"]:
    baseline_vals = np.array(all_final_hvs[baseline])
    diff = hsevo_vals - baseline_vals
    
    # Wilcoxon signed-rank test
    try:
        stat, p_val = stats.wilcoxon(hsevo_vals, baseline_vals, alternative='two-sided')
    except ValueError as e:
        stat, p_val = 0, 1.0
        
    # Cliff's delta
    n = len(hsevo_vals)
    count_greater = sum(1 for i in range(n) for j in range(n) if hsevo_vals[i] > baseline_vals[j])
    count_less = sum(1 for i in range(n) for j in range(n) if hsevo_vals[i] < baseline_vals[j])
    cliffs_d = (count_greater - count_less) / (n * n)
    
    # Effect size interpretation
    abs_d = abs(cliffs_d)
    if abs_d < 0.147:
        effect = "negligible"
    elif abs_d < 0.33:
        effect = "small"
    elif abs_d < 0.474:
        effect = "medium"
    else:
        effect = "large"
    
    sig = "YES" if p_val < 0.05 else "NO"
    print(f"  HSEvo vs {baseline:>8}: W={stat:>6.1f}, p={p_val:.4f} (sig={sig}), Cliff's δ={cliffs_d:+.3f} ({effect})")

# 4. Cross-dataset std dev (robustness)
print("\n\n--- ROBUSTNESS: Cross-Dataset Std Dev ---")
for a in algs:
    print(f"  {a:>8}: σ = {np.std(all_final_hvs[a]):.4f}")

# 5. Step-by-step timeseries average (for plot)
print("\n\n--- AVERAGE HV PER STEP (for plot) ---")
avg_ts = {a: np.zeros(10) for a in algs}
for ds in datasets:
    for a in algs:
        for s in seeds:
            ts = np.array(data[ds][a][s]["hv_timeseries"])
            avg_ts[a] += ts
num = len(datasets) * len(seeds)
for a in algs:
    avg_ts[a] /= num
    
print(f"{'Step':<6}", end="")
for a in algs:
    print(f"  {a:>8}", end="")
print()
for step in range(10):
    print(f"t={step+1:<4}", end="")
    for a in algs:
        print(f"  {avg_ts[a][step]:>8.4f}", end="")
    print()

# 6. API calls estimate
print("\n\n--- COST: HSEvo API Calls ---")
# pop_size=6, init_pop_size=6, max_fe=12, max_iter=2
# Each evaluation = 1 API call. init_pop (6) + 12 max_fe * 2 iterations = 30 max API calls
print("  pop_size=6, max_fe=12, max_iter=2")
print("  Estimated API calls per dataset: ~30 (6 init + 12*2 evolution)")
total_time = sum(all_times["hsevo"])
print(f"  Total HSEvo wall-clock time (10 datasets × 3 seeds): {total_time:.1f}s = {total_time/60:.1f} min")
avg_per_ds = total_time / (len(datasets) * len(seeds))
print(f"  Average wall-clock per run: {avg_per_ds:.1f}s")

