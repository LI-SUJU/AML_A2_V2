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
from constants import DATA_FILES, NUM_ITERATIONS, NUM_EXPERIMENTS, MIN_ANCHOR_SIZES

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--performance_file', type=str, default='config_performances_dataset-6.csv')
    parser.add_argument('--num_iterations', type=int, default=10)
    return parser.parse_args()

def calculate_evaluations(evaluation_dict, weight_function):
    return sum(weight_function(anchor) * evaluations for anchor, evaluations in evaluation_dict.items())

def calculate_linear_evaluations(evaluation_dict):
    return calculate_evaluations(evaluation_dict, lambda x: x)

def calculate_log_evaluations(evaluation_dict):
    return calculate_evaluations(evaluation_dict, np.log)

def calculate_square_evaluations(evaluation_dict):
    return calculate_evaluations(evaluation_dict, lambda x: x**2)

def perform_experiment(vertical_evaluator, iterations, config_space, dataset_name):
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
    plt.savefig(f'{dataset_name}_{vertical_evaluator.method}_{iterations}.png')
    plt.close()

    linear_evaluations = calculate_linear_evaluations(evaluation_counts)
    log_evaluations = calculate_log_evaluations(evaluation_counts)
    square_evaluations = calculate_square_evaluations(evaluation_counts)
    return linear_evaluations, log_evaluations, square_evaluations, best_result

def run_experiment(config_space, data_file, num_iterations, num_experiments, min_anchor_size):
    dataset_name = data_file.split('_')[-1][:-4]
    data_frame = pd.read_csv(data_file)
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(data_frame)
    anchors = sorted(data_frame.loc[data_frame.anchor_size >= min_anchor_size, 'anchor_size'].unique())
    print('Surrogate model fitted')

    results = {'LCCV': {'linear': [], 'log': [], 'square': [], 'best': []},
               'IPL': {'linear': [], 'log': [], 'square': [], 'best': []}}

    for method in ['LCCV', 'IPL']:
        vertical_evaluator = LCCV(surrogate_model, anchors) if method == 'LCCV' else IPL(surrogate_model, anchors)
        for _ in range(num_experiments):
            linear_evaluations, log_evaluations, square_evaluations, best_score = perform_experiment(vertical_evaluator, num_iterations, config_space, dataset_name)
            results[method]['linear'].append(linear_evaluations)
            results[method]['log'].append(log_evaluations)
            results[method]['square'].append(square_evaluations)
            results[method]['best'].append(best_score)
        print(f'{method} done')

    ratio_linear = (1 / np.array(results['LCCV']['best']) / np.array(results['LCCV']['linear'])) / \
                   (1 / np.array(results['IPL']['best']) / np.array(results['IPL']['linear']))
    ratio_log = (1 / np.array(results['LCCV']['best']) / np.array(results['LCCV']['log'])) / \
                (1 / np.array(results['IPL']['best']) / np.array(results['IPL']['log']))
    ratio_square = (1 / np.array(results['LCCV']['best']) / np.array(results['LCCV']['square'])) / \
                   (1 / np.array(results['IPL']['best']) / np.array(results['IPL']['square']))

    return {
        ('LCCV', 'score'): f'{np.mean(results["LCCV"]["best"]):.2f} std {np.std(results["LCCV"]["best"]):.3f}',
        ('LCCV', 'evaluations_linear'): f'{np.mean(results["LCCV"]["linear"]):.2f} std {np.std(results["LCCV"]["linear"]):.3f}',
        ('LCCV', 'evaluations_log'): f'{np.mean(results["LCCV"]["log"]):.2f} std {np.std(results["LCCV"]["log"]):.3f}',
        ('LCCV', 'evaluations_square'): f'{np.mean(results["LCCV"]["square"]):.2f} std {np.std(results["LCCV"]["square"]):.3f}',
        ('IPL', 'score'): f'{np.mean(results["IPL"]["best"]):.2f} std {np.std(results["IPL"]["best"]):.3f}',
        ('IPL', 'evaluations_linear'): f'{np.mean(results["IPL"]["linear"]):.2f} std {np.std(results["IPL"]["linear"]):.3f}',
        ('IPL', 'evaluations_log'): f'{np.mean(results["IPL"]["log"]):.2f} std {np.std(results["IPL"]["log"]):.3f}',
        ('IPL', 'evaluations_square'): f'{np.mean(results["IPL"]["square"]):.2f} std {np.std(results["IPL"]["square"]):.3f}',
        ('ratio', 'evaluations_linear'): f'{np.mean(ratio_linear):.2f} std {np.std(ratio_linear):.3f}',
        ('ratio', 'evaluations_log'): f'{np.mean(ratio_log):.2f} std {np.std(ratio_log):.3f}',
        ('ratio', 'evaluations_square'): f'{np.mean(ratio_square):.2f} std {np.std(ratio_square):.3f}'
    }

def format_results_as_table(result_dataframe):
    return tabulate(result_dataframe, headers='keys', tablefmt='grid', showindex="always")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    config_space = ConfigSpace.ConfigurationSpace.from_json('lcdb_config_space_knn.json')
    data_files = ['config_performances_dataset-6.csv', 'config_performances_dataset-11.csv', 'config_performances_dataset-1457.csv']
    num_iterations = 100
    num_experiments = 10
    min_anchor_sizes = {'dataset-6': 16, 'dataset-11': 32, 'dataset-1457': 128}

    metric_tuples = [(method, metric) for method, metric in product(['LCCV', 'IPL', 'ratio'], ['score', 'evaluations_linear', 'evaluations_log', 'evaluations_square'])]
    index = pd.MultiIndex.from_tuples(metric_tuples, names=['method', 'metric'])
    columns = [data_file.split('_')[-1][:-4] for data_file in data_files]
    result_dataframe = pd.DataFrame(columns=columns, index=index)

    for data_file, dataset_name in zip(data_files, columns):
        print(dataset_name)
        result = run_experiment(config_space, data_file, num_iterations, num_experiments, min_anchor_sizes[dataset_name])
        result_dataframe[dataset_name] = result
        result_dataframe.to_csv('comparison_results_with_ratio.csv')

    formatted_table = format_results_as_table(result_dataframe)
    with open('formatted_comparison_results_with_ratio.txt', 'w') as file:
        file.write(formatted_table)
