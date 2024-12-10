import logging
import numpy as np
import typing
import pandas as pd
from scipy.optimize import curve_fit
from vertical_model_evaluator import VerticalModelEvaluator


class IPL(VerticalModelEvaluator):
    
    @staticmethod
    def extrapolate_performance(results: typing.List[typing.Tuple[int, float]], target_anchor: int) -> float:
        """
        Performs performance extrapolation using a power-law function.

        :param results: List of tuples containing anchor sizes and corresponding performances
        :param target_anchor: The anchor size at which to extrapolate the performance
        :return: The extrapolated performance at the target anchor size
        """
        def power_law_function(x, m, c, c0):
            return c0 + x**m * c
        
        anchor_sizes = np.array([x for x, _ in results])
        performances = np.array([y for _, y in results])
        popt, _ = curve_fit(power_law_function, anchor_sizes, performances, maxfev=1000000)
        extrapolated_performance = power_law_function(target_anchor, *popt)
        return extrapolated_performance
    
    def evaluate_model(self, best_performance_so_far: typing.Optional[float], configuration: typing.Dict, evaluations_count: typing.Dict) -> typing.List[typing.Tuple[int, float]]:
        """
        Performs staged evaluation of the model on increasing anchor sizes.
        Determines an optimistic extrapolation after each evaluation. Stops
        the evaluation if the extrapolation cannot improve over the best
        performance so far. If the best performance is not determined (None),
        it evaluates immediately on the final anchor.

        :param best_performance_so_far: The best performance obtained so far
        :param configuration: A dictionary indicating the configuration
        :param evaluations_count: A dictionary to keep track of the number of evaluations per anchor size
        :return: A list of tuples containing anchor sizes and estimated performances
        """
        self.method = 'IPL'
        
        if best_performance_so_far is None:
            configuration["anchor_size"] = self.final_anchor
            evaluations_count[self.final_anchor] += 1
            config_df = pd.DataFrame([dict(configuration)])
            performance = self.surrogate_model.predict(config_df)[0]
            return [(self.final_anchor, performance)], evaluations_count
        
        evaluation_results = []
        for anchor in self.anchors: 
            configuration["anchor_size"] = anchor
            config_df = pd.DataFrame([dict(configuration)])
            evaluations_count[anchor] += 1
            performance = self.surrogate_model.predict(config_df)[0]
            evaluation_results.append((anchor, performance))
            
            if len(evaluation_results) >= 3: 
                extrapolated_performance = self.extrapolate_performance(evaluation_results, self.final_anchor)
                if extrapolated_performance > best_performance_so_far:
                    break
                
        return evaluation_results, evaluations_count
