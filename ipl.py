import typing
import pandas as pd
from scipy.optimize import curve_fit
from vertical_model_evaluator import VerticalModelEvaluator

class IPL(VerticalModelEvaluator):
    
    @staticmethod
    def extrapolate_performance(results: typing.List[typing.Tuple[int, float]], target_anchor: int) -> float:
        def power_law_function(x, m, c, c0):
            return c0 + x**m * c
        
        anchor_sizes, performances = zip(*results)
        popt, _ = curve_fit(power_law_function, anchor_sizes, performances, maxfev=1000000)
        return power_law_function(target_anchor, *popt)
    
    def evaluate_model(self, best_performance_so_far: typing.Optional[float], configuration: typing.Dict, evaluations_count: typing.Dict) -> typing.List[typing.Tuple[int, float]]:
        self.method = 'IPL'
        
        if best_performance_so_far is None:
            configuration["anchor_size"] = self.final_anchor
            evaluations_count[self.final_anchor] += 1
            performance = self.surrogate_model.predict(pd.DataFrame([configuration]))[0]
            return [(self.final_anchor, performance)], evaluations_count
        
        evaluation_results = []
        for anchor in self.anchors: 
            configuration["anchor_size"] = anchor
            evaluations_count[anchor] += 1
            performance = self.surrogate_model.predict(pd.DataFrame([configuration]))[0]
            evaluation_results.append((anchor, performance))
            
            if len(evaluation_results) >= 3: 
                extrapolated_performance = self.extrapolate_performance(evaluation_results, self.final_anchor)
                if extrapolated_performance > best_performance_so_far:
                    break
                
        return evaluation_results, evaluations_count
