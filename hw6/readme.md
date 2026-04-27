# SE & AI Group 15 — Project Replication

This repository contains the complete replication package for the paper:  
**"Evolutionary LLMs for Automated Heuristic Generation in Multi-Objective Software Engineering Optimization"**

## 🚀 Reproduction Instructions

### 1. Prerequisites
- **Python**: 3.13+
- **Hardware**: Tested on M4 MacBook Air (24GB RAM)
- **API Key**: An OpenAI API key for `gpt-4o-mini-2024-07-18`. Set it as an environment variable:
  ```bash
  export OPENAI_API_KEY='your-key-here'
  ```

### 2. Installation
Navigate to the `HSEvo` directory and install the required dependencies:
```bash
cd HSEvo
pip install -r requirements.txt
```

### 3. Run Experiments
To replicate the full experimental pipeline across all 10 MOOT datasets with 3 independent seeds and a 100-step evaluation budget, execute:
```bash
python run_experiments.py --datasets all --seeds 42 123 7 --budget 100 --pop_size 6
```
This script will:
- Evaluate 5 algorithms: **Random**, **Greedy**, **MOTPE**, **NSGA-II**, and **HSEvo**.
- Store step-by-step Hypervolume (HV) scores in `experiment_results.json`.
- Log per-dataset runtimes and API usage.
- Expected runtime: ~9.0 minutes.

### 4. Generate Figures
Once the experiments are complete, regenerate Figure 2 (Warm-Start) and Figure 3 (Full Convergence) using:
```bash
python plot_results.py
```
The resulting figures will be saved as `crossover_analysis.png` and `fig3_full_convergence.pdf`.

---

## 📂 Repository Structure
- `HSEvo/`: Main framework and experiment scripts.
  - `run_experiments.py`: The entry point for the evaluation pipeline.
  - `plot_results.py`: Visualization script for HV convergence.
  - `experiment_results.json`: Raw results from 150 experimental runs.
  - `software.bib`: Bibliography of 150+ related works.
- `context/`: Search terms and raw bibliometric data from Publish or Perish.
- `Knee Papers/`: PDFs and notes on the 18 core papers identified via the citation knee heuristic.
