import argparse
import ConfigSpace
import logging
import matplotlib.pyplot as plt
import pandas as pd
from lccv import LCCV
from ipl import IPL
from surrogate_model import SurrogateModel
import numpy as np
from itertools import product
from matplotlib.colors import ListedColormap
from tabulate import tabulate
from constants import DATA_FILES, NUM_ITERATIONS, NUM_EXPERIMENTS, MIN_ANCHOR_SIZES, CONFIG_SPACE
from tools import calculate_linear_evaluations, calculate_log_evaluations, calculate_square_evaluations
import warnings
from scipy.optimize import OptimizeWarning

def custom_filter():
    # Ignore specific FutureWarnings
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message="Downcasting object dtype arrays on .fillna, .*"
    )
    
    # Ignore OptimizeWarning
    warnings.filterwarnings("ignore", category=OptimizeWarning)

    warnings.filterwarnings("ignore", category=UserWarning)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--performance_file', type=str, default='config_performances_dataset-6.csv')
    parser.add_argument('--num_iterations', type=int, default=10)
    return parser.parse_args()


def perform_single_experiment(vertical_evaluator, iterations, config_space, dataset_name):
    evaluation_counts = {anchor: 0 for anchor in vertical_evaluator.anchors}
    best_result = None

    for iteration in range(iterations):
        new_configuration = dict(config_space.sample_configuration())
        results, evaluation_counts = vertical_evaluator.evaluate_model(best_result, new_configuration, evaluation_counts)
        final_result = results[-1][1]
        if best_result is None or final_result < best_result:
            best_result = final_result

        x_values = [result[0] for result in results]
        y_values = [result[1] for result in results]
        norm = plt.Normalize(0, iterations)
        cmap = plt.get_cmap('plasma', iterations)
        colors = cmap(norm(iteration))
        listed_cmap = ListedColormap(cmap(np.linspace(0, 1, iterations)))
        plt.plot(x_values, y_values, color=colors, linestyle='-', linewidth=0.5)
        plt.scatter(x_values, y_values, c=[colors] * len(x_values), cmap=listed_cmap, s=10)

    plt.title(f'{vertical_evaluator.method} on {dataset_name}')
    plt.ylabel('Error')
    plt.xlabel('Anchor Size')
    plt.axhline(best_result, color='black', linestyle='--', linewidth=0.5)
    plt.text(0, best_result, f'Best so far: {best_result:.2f}', color='black')
    plt.savefig(f'./plots/{vertical_evaluator.method}_{dataset_name}.png')
    plt.close()

    linear_evaluations = calculate_linear_evaluations(evaluation_counts)
    log_evaluations = calculate_log_evaluations(evaluation_counts)
    square_evaluations = calculate_square_evaluations(evaluation_counts)
    return linear_evaluations, log_evaluations, square_evaluations, best_result


def run_experiments(config_space, data_file, num_iterations, num_experiments, min_anchor_size):
    dataset_name = data_file.split('_')[-1][:-4]
    data_frame = pd.read_csv(data_file)
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(data_frame)
    anchors = sorted(data_frame.loc[data_frame.anchor_size >= min_anchor_size, 'anchor_size'].unique())
    # logging.info(f'Surrogate model fitted for dataset: {dataset_name}')

    results = {'LCCV': {'linear': [], 'log': [], 'square': [], 'best': []},
               'IPL': {'linear': [], 'log': [], 'square': [], 'best': []}}

    for method in ['LCCV', 'IPL']:
        logging.info(f'Starting experiments for {method} on dataset: {dataset_name}')
        vertical_evaluator = LCCV(surrogate_model, anchors) if method == 'LCCV' else IPL(surrogate_model, anchors)
        for exp in range(1, num_experiments + 1):
            logging.info(f'Running experiment {exp}/{num_experiments} for {method} on {dataset_name}')
            linear_evaluations, log_evaluations, square_evaluations, best_score = perform_single_experiment(vertical_evaluator, num_iterations, config_space, dataset_name)
            results[method]['linear'].append(linear_evaluations)
            results[method]['log'].append(log_evaluations)
            results[method]['square'].append(square_evaluations)
            results[method]['best'].append(best_score)
        logging.info(f'{method} experiments completed for dataset: {dataset_name}')

    ratio_linear = (1 / np.array(results['LCCV']['best']) / np.array(results['LCCV']['linear'])) / \
                   (1 / np.array(results['IPL']['best']) / np.array(results['IPL']['linear']))
    ratio_log = (1 / np.array(results['LCCV']['best']) / np.array(results['LCCV']['log'])) / \
                (1 / np.array(results['IPL']['best']) / np.array(results['IPL']['log']))
    ratio_square = (1 / np.array(results['LCCV']['best']) / np.array(results['LCCV']['square'])) / \
                   (1 / np.array(results['IPL']['best']) / np.array(results['IPL']['square']))

    return {
        ('LCCV', 'error'): f'{np.mean(results["LCCV"]["best"]):.2f} ± {np.std(results["LCCV"]["best"]):.3f}',
        ('LCCV', 'evaluations_linear'): f'{np.mean(results["LCCV"]["linear"]):.2f} ± {np.std(results["LCCV"]["linear"]):.3f}',
        ('LCCV', 'evaluations_log'): f'{np.mean(results["LCCV"]["log"]):.2f} ± {np.std(results["LCCV"]["log"]):.3f}',
        ('LCCV', 'evaluations_square'): f'{np.mean(results["LCCV"]["square"]):.2f} ± {np.std(results["LCCV"]["square"]):.3f}',
        ('IPL', 'error'): f'{np.mean(results["IPL"]["best"]):.2f} ± {np.std(results["IPL"]["best"]):.3f}',
        ('IPL', 'evaluations_linear'): f'{np.mean(results["IPL"]["linear"]):.2f} ± {np.std(results["IPL"]["linear"]):.3f}',
        ('IPL', 'evaluations_log'): f'{np.mean(results["IPL"]["log"]):.2f} ± {np.std(results["IPL"]["log"]):.3f}',
        ('IPL', 'evaluations_square'): f'{np.mean(results["IPL"]["square"]):.2f} ± {np.std(results["IPL"]["square"]):.3f}',
        ('Ratio', 'evaluations_linear'): f'{np.mean(ratio_linear):.2f} ± {np.std(ratio_linear):.3f}',
        ('Ratio', 'evaluations_log'): f'{np.mean(ratio_log):.2f} ± {np.std(ratio_log):.3f}',
        ('Ratio', 'evaluations_square'): f'{np.mean(ratio_square):.2f} ± {np.std(ratio_square):.3f}'
    }


def format_results_as_table_v3(result_dataframe):
    datasets = result_dataframe.columns
    stats = ['Error', 'Linear Evaluations', 'Log Evaluations', 'Square Evaluations']
    methods = ['LCCV', 'IPL', 'Ratio']
    formatted_data = []

    for dataset in datasets:
        for metric, stat in zip(['error', 'evaluations_linear', 'evaluations_log', 'evaluations_square'], stats):
            lccv_value = result_dataframe.loc[('LCCV', metric), dataset]
            ipl_value = result_dataframe.loc[('IPL', metric), dataset]
            ratio_value = result_dataframe.loc[('Ratio', metric), dataset] if metric != 'error' else '-'

            formatted_data.append([dataset, stat, lccv_value, ipl_value, ratio_value])

    columns = ['Dataset', 'Stats', 'LCCV Mean ± Std', 'IPL Mean ± Std', 'Ratio Mean ± Std']
    return pd.DataFrame(formatted_data, columns=columns)


def initialize_dataframe(data_files):
    metric_tuples = [(method, metric) for method, metric in product(['LCCV', 'IPL', 'Ratio'], ['error', 'evaluations_linear', 'evaluations_log', 'evaluations_square'])]
    index = pd.MultiIndex.from_tuples(metric_tuples, names=['Method', 'Metric'])
    columns = [data_file.split('_')[-1][:-4] for data_file in data_files]
    return pd.DataFrame(columns=columns, index=index)


def run_record_experiments(data_files, config_space, num_iterations, num_experiments, min_anchor_sizes):
    result_dataframe = initialize_dataframe(data_files)
    for data_file, dataset_name in zip(data_files, result_dataframe.columns):
        logging.info(f'Processing dataset: {dataset_name}')
        result = run_experiments(config_space, data_file, num_iterations, num_experiments, min_anchor_sizes[dataset_name])
        result_dataframe[dataset_name] = result
        result_dataframe.to_csv('./results/results_with_ratio.csv')
        logging.info(f'Dataset {dataset_name} processing completed.')
    return result_dataframe


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    custom_filter()

    config_space = CONFIG_SPACE
    data_files = DATA_FILES
    num_iterations = NUM_ITERATIONS
    num_experiments = NUM_EXPERIMENTS
    min_anchor_sizes = MIN_ANCHOR_SIZES

    result_dataframe = run_record_experiments(data_files, config_space, num_iterations, num_experiments, min_anchor_sizes)
    formatted_table_df = format_results_as_table_v3(result_dataframe)
    formatted_table_df.to_csv('./results/formatted_results_with_ratio.csv')
    print(tabulate(formatted_table_df, headers='keys', tablefmt='github'))
