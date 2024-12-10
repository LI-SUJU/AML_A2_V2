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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='config_performances_dataset-6.csv')
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--num_iterations', type=int, default=10)

    return parser.parse_args()

def count_evalutions_linear(eval_dict):
    ''' Calculate total number of evaluations weighted by anchor size.
    
    param: eval_dict (dict): Dictionary mapping anchor sizes to number of evaluations
    
    return: int: Total weighted number of evaluations
    '''
    
    evaluations = 0
    for anchor,evals in eval_dict.items():
        evaluations+=anchor*evals
    return evaluations

def count_evaluations_log(eval_dict):
    ''' Calculate total number of evaluations weighted by log(anchor size).
    
    param: eval_dict (dict): Dictionary mapping anchor sizes to number of evaluations
    
    return: int: Total weighted number of evaluations
    '''
    evaluations = 0
    for anchor,evals in eval_dict.items():
        evaluations+=np.log(anchor)*evals
    return evaluations

def count_evaluations_square(eval_dict):
    ''' Calculate total number of evaluations weighted by square root of anchor size.
    
    param: eval_dict (dict): Dictionary mapping anchor sizes to number of evaluations
    
    return: int: Total weighted number of evaluations
    '''
    evaluations = 0
    for anchor,evals in eval_dict.items():
        evaluations += anchor**2 * evals
    return evaluations

def experiment(vertical_eval,iterations,config_space,name):
    '''
    Run a single experiment with given vertical evaluator.
    
    param: vertical_eval: LCCV or IPL evaluator instance
    param: iterations (int): number of configurations to evaluate
    param: config_space: configSpace containing hyperparameter definitions
    param: name (str): name of dataset for plot labeling
    
    return: tuple: (total_evaluations, best_score_found)
    
    '''
    evaluations_dict = {anchor:0 for anchor in vertical_eval.anchors}
    best_so_far = None
    
    for _ in range(iterations):
        theta_new = dict(config_space.sample_configuration())
        result,evaluations_dict = vertical_eval.evaluate_model(best_so_far, theta_new,evaluations_dict)
        final_result = result[-1][1]
        if best_so_far is None or final_result < best_so_far:
            best_so_far = final_result
        x_values = [i[0] for i in result]
        y_values = [i[1] for i in result]
        plt.plot(x_values, y_values, "-o")
    plt.title(f'{name} {vertical_eval.method}')
    plt.ylabel('error')
    plt.xlabel('anchor')
    plt.savefig(f'{name}_{vertical_eval.method}_{iterations}.png')
    plt.close()
    evalutations_linear = count_evalutions_linear(evaluations_dict)
    evalutations_log = count_evaluations_log(evaluations_dict)
    evalutations_square = count_evaluations_square(evaluations_dict)
    return evalutations_linear, evalutations_log, evalutations_square, best_so_far


def run(config_space,data_file,nr_iterations,nr_experiments,min_anchor):
    '''
    Run complete experiment comparing LCCV and IPL methods.
    '''
    name = data_file.split('_')[-1][:-4]
    df = pd.read_csv(data_file)
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df)
    anchors = sorted(df.loc[df.anchor_size>=min_anchor,'anchor_size'].unique())
    print('surrogate model fitted')

    # LCCV
    vertical_eval = LCCV(surrogate_model, anchors)
    lccv_eval_linear = []
    lccv_eval_log = []
    lccv_eval_square = []
    lccv_best = []
    for _ in range(nr_experiments):
        evalutations_linear, evalutations_log, evalutations_square,best_score = experiment(vertical_eval,nr_iterations,config_space,name)
        lccv_eval_linear.append(evalutations_linear)
        lccv_eval_log.append(evalutations_log)
        lccv_eval_square.append(evalutations_square)
        lccv_best.append(best_score)
    print('LCCV done')
    # IPL
    ipl_eval_linear = []
    ipl_eval_log = []
    ipl_eval_square = []
    ipl_best = []
    vertical_eval = IPL(surrogate_model, anchors)
    for i in range(nr_experiments):
        evalutations_linear, evalutations_log, evalutations_square,best_score = experiment(vertical_eval,nr_iterations,config_space,name)
        ipl_eval_linear.append(evalutations_linear)
        ipl_eval_log.append(evalutations_log)
        ipl_eval_square.append(evalutations_square)
        ipl_best.append(best_score)
    print('IPL done')
    
    # ratio of lccv to ipl: lccv/ipl
    ratio_linear = ((1/np.array(lccv_best))/np.array(lccv_eval_linear))/((1/np.array(ipl_best))/np.array(ipl_eval_linear))
    ratio_log = ((1/np.array(lccv_best))/np.array(lccv_eval_log))/((1/np.array(ipl_best))/np.array(ipl_eval_log))
    ratio_square = ((1/np.array(lccv_best))/np.array(lccv_eval_square))/((1/np.array(ipl_best))/np.array(ipl_eval_square))
    # Gather results
    # lccv_eval = np.array(lccv_eval)/anchors[-1]
    # ipl_eval = np.array(ipl_eval)/anchors[-1]
    return {('LCCV','score'):f'{np.mean(lccv_best):.2f} std {np.std(lccv_best):.3f}',
            ('LCCV','evalutations_linear'):f'{np.mean(lccv_eval_linear):.2f} std {np.std(lccv_eval_linear):.3f}',
            ('LCCV','evalutations_log'):f'{np.mean(lccv_eval_log):.2f} std {np.std(lccv_eval_log):.3f}',
            ('LCCV','evalutations_square'):f'{np.mean(lccv_eval_square):.2f} std {np.std(lccv_eval_square):.3f}',
            ('IPL','score'):f' {np.mean(ipl_best):.2f} std {np.std(ipl_best):.3f}',
            ('IPL','evalutations_linear'):f'{np.mean(ipl_eval_linear):.2f} std {np.std(ipl_eval_linear):.3f}',
            ('IPL','evalutations_log'):f'{np.mean(ipl_eval_log):.2f} std {np.std(ipl_eval_log):.3f}',
            ('IPL','evalutations_square'):f'{np.mean(ipl_eval_square):.2f} std {np.std(ipl_eval_square):.3f}',
            ('ratio','evalutations_linear'):f'{np.mean(ratio_linear):.2f} std {np.std(ratio_linear):.3f}',
            ('ratio','evalutations_log'):f'{np.mean(ratio_log):.2f} std {np.std(ratio_log):.3f}',
            ('ratio','evalutations_square'):f'{np.mean(ratio_square):.2f} std {np.std(ratio_square):.3f}'
            }


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    
    config_space = ConfigSpace.ConfigurationSpace.from_json('lcdb_config_space_knn.json')
    data_files = ['config_performances_dataset-6.csv',
                  'config_performances_dataset-11.csv',
                  'config_performances_dataset-1457.csv']
    nr_iterations = 100
    nr_experiments = 10
    data_file_names = [data_file.split('_')[-1][:-4] for data_file in data_files]
    min_anchor = {'dataset-6':16,
                  'dataset-11':32,
                  'dataset-1457':128}
    tuples = [(a,b) for a,b in product(['LCCV','IPL','ratio'],['score','evalutations_linear','evalutations_log','evalutations_square'])]
    index = pd.MultiIndex.from_tuples(tuples,names=['method','metric'])
    columns = data_file_names
    result_df = pd.DataFrame(columns=columns ,index=index)

    for data_file,name in zip(data_files,columns):
        print(name)
        result = run(config_space,data_file,nr_iterations,nr_experiments,min_anchor[name])
        result_df[name] = result 
        result_df.to_csv('comparison_results_with_ratio.csv')

    def format_results_as_table(result_df):
        """
        Format the result DataFrame into a clean, tabular format using the `tabulate` library.
        """
        formatted_table = tabulate(result_df, headers='keys', tablefmt='grid', showindex="always")
        return formatted_table
    from tabulate import tabulate
    formatted_table = format_results_as_table(result_df)
    with open('formatted_comparison_results_with_ratio.txt', 'w') as f:
        f.write(formatted_table)