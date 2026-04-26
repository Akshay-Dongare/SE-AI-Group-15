import json
import matplotlib.pyplot as plt
import numpy as np

with open('experiment_results_timeseries.json', 'r') as f:
    results = json.load(f)

# Average across all datasets
datasets = list(results.keys())
algorithms = list(results[datasets[0]].keys())
steps = len(results[datasets[0]][algorithms[0]])

avg_results = {alg: np.zeros(steps) for alg in algorithms}

for alg in algorithms:
    valid_datasets = 0
    for ds in datasets:
        if results[ds].get(alg) is not None:
            avg_results[alg] += np.array(results[ds][alg])
            valid_datasets += 1
    if valid_datasets > 0:
        avg_results[alg] /= valid_datasets

plt.figure(figsize=(10, 6))
for alg in algorithms:
    plt.plot(range(1, steps + 1), avg_results[alg], marker='o', label=alg.upper())

plt.title('Budget Sensitivity and Crossover Analysis (Average Hypervolume)')
plt.xlabel('Active-Learning Step (Evaluation Budget)')
plt.ylabel('Average Hypervolume')
plt.grid(True)
plt.legend()
plt.savefig('crossover_analysis.png', dpi=300)
print("Plot saved to crossover_analysis.png")
