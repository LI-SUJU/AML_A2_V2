# Experiment Guidelines

This README provides detailed instructions for setting up and running the experiment described in this repository. The code includes configurations for comparing two methods, LCCV and IPL, using a surrogate model to simulate learning curves. Below, we explain the process step-by-step.

## Environment Setup
Before running the experiment, ensure that all necessary libraries and dependencies are installed. The key dependencies include:

- `ConfigSpace` for handling configuration spaces.
- `pandas`, `numpy`, and `matplotlib` for data manipulation, numerical computation, and plotting.
- `lccv`, `ipl`, and `SurrogateModel`, which are specific to this experiment.
- `tabulate` for formatting result tables.
- `scipy` for optimization warnings.

To avoid warnings cluttering the output, a custom filter is implemented to suppress `FutureWarnings`, `UserWarnings`, and `OptimizeWarnings`.

## Running a Single Experiment
The function `perform_single_experiment` evaluates configurations using LCCV or IPL:

1. **Initialization:** Anchor sizes are initialized as an increasing sequence (e.g., [16, 32, 64, 128]). Evaluation counts are tracked for each anchor size.
2. **Configuration Sampling:** A configuration is randomly sampled from the configuration space.
3. **Evaluation at Anchors:** The sampled configuration is sequentially evaluated at increasing anchor sizes. The methods are applied as follows:
   - **LCCV:** Optimistic extrapolation predicts performance beyond the observed anchor points.
   - **IPL:** A parametric Inverse Power Law model is fitted to the observed performance to extrapolate results.
4. **Performance Tracking:** The best observed performance is recorded, and configurations performing worse than the best-so-far are discarded.
5. **Plotting:** Results are visualized as learning curves, saved as PNG files, and annotated with the best performance.

## Running Experiments Across Datasets
The function `run_experiments` automates the process for multiple datasets:

1. **Dataset Preparation:** Each dataset is read from a CSV file. Anchor sizes are determined based on a predefined minimum size.
2. **Surrogate Model:** A surrogate model is trained on the dataset to simulate learning curve behavior.
3. **Comparison of Methods:** LCCV and IPL are applied to the dataset. For each method, the following metrics are computed:
   - Error (mean squared error).
   - Computational costs weighted by linear, logarithmic, and quadratic complexities.
   - Efficiency ratios comparing LCCV and IPL.

## Storing and Formatting Results
Results are stored in a structured format and saved as CSV files. A function `format_results_as_table_v3` generates a readable summary with the following columns:

- Dataset.
- Metric (e.g., error, evaluations with different complexities).
- Results for LCCV, IPL, and their efficiency ratios.

The formatted table is saved as `formatted_results_with_ratio.csv`.

## Running the Code
**To execute the experiments and get the results shown in the report, run the script directly:**

```bash
python run_experiments.py
```

## Output
The script generates the following outputs:

- **Plots:** Learning curves for each method and dataset, saved as PNG files in the `plots/` directory.
- **Results:** Detailed results in CSV format (`results_with_ratio.csv`) and a formatted summary (`formatted_results_with_ratio.csv`).
- **Logs:** Informative logs indicating the progress and completion of experiments.

