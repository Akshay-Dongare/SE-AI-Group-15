import json
import matplotlib.pyplot as plt
import numpy as np

# Use the correct renamed results file
with open('experiment_results.json', 'r') as f:
    data = json.load(f)

datasets = list(data.keys())
algs = ["random", "greedy", "motpe", "nsgaii", "hsevo"]
seeds = ["42", "123", "7"]
labels = {"random": "Random", "greedy": "Greedy", "motpe": "MOTPE", "nsgaii": "NSGA-II", "hsevo": "HSEvo (LLM)"}
colors = {"random": "#888888", "greedy": "#e74c3c", "motpe": "#3498db", "nsgaii": "#2ecc71", "hsevo": "#9b59b6"}
markers = {"random": "s", "greedy": "^", "motpe": "D", "nsgaii": "o", "hsevo": "P"}

# Average HV per step across all datasets and seeds
avg_ts = {a: np.zeros(100) for a in algs}
for ds in datasets:
    for a in algs:
        for s in seeds:
            ts = np.array(data[ds][a][s]["hv_timeseries"])
            avg_ts[a] += ts
num = len(datasets) * len(seeds)
for a in algs:
    avg_ts[a] /= num

# ─── Figure 3: Full 100-step Convergence ───
fig, ax = plt.subplots(figsize=(10, 6))
for a in algs:
    ax.plot(range(1, 101), avg_ts[a], marker=markers[a], label=labels[a], 
            color=colors[a], linewidth=2, markersize=4, markevery=5)

ax.set_title('Full Budget Convergence Analysis\n(Average Hypervolume vs. Active-Learning Step, pop_size=6, 3 seeds)', 
             fontsize=13, fontweight='bold')
ax.set_xlabel('Active-Learning Step (Evaluation Budget)', fontsize=12)
ax.set_ylabel('Average Hypervolume', fontsize=12)
ax.set_xticks(np.arange(0, 101, 10))
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='lower right')
ax.set_ylim(0.55, 1.0)

plt.tight_layout()
plt.savefig('fig3_full_convergence.pdf', dpi=300)
plt.savefig('crossover_analysis.png', dpi=300)
print("Plot saved to fig3_full_convergence.pdf and crossover_analysis.png")

# ─── Figure 2: 10-step Warm-Start Zoom ───
fig2, ax2 = plt.subplots(figsize=(8, 5))
for a in algs:
    ax2.plot(range(1, 11), avg_ts[a][:10], marker=markers[a], label=labels[a], 
             color=colors[a], linewidth=2, markersize=6)

ax2.set_title('Zero-Shot Warm-Start Analysis (Initial 10 Steps)\nLLM Semantic Prior Advantage', 
              fontsize=13, fontweight='bold')
ax2.set_xlabel('Active-Learning Step', fontsize=12)
ax2.set_ylabel('Average Hypervolume', fontsize=12)
ax2.set_xticks(range(1, 11))
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11, loc='lower right')
ax2.set_ylim(0.55, 0.9)

plt.tight_layout()
plt.savefig('fig2_warm_start_zoom.pdf', dpi=300)
print("Plot saved to fig2_warm_start_zoom.pdf")
