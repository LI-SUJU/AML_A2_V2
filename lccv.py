import typing
import pandas as pd
from vertical_model_evaluator import VerticalModelEvaluator

class LCCV(VerticalModelEvaluator):
    
    @staticmethod
    def optimistic_extrapolation(prev_anchor: int, prev_performance: float, 
                                 curr_anchor: int, curr_performance: float, 
                                 target_anchor: int) -> float:
        slope = (prev_performance - curr_performance) / (prev_anchor - curr_anchor)
        return curr_performance + (target_anchor - curr_anchor) * slope
    
    def evaluate_model(self, best_performance: typing.Optional[float], 
                       config: typing.Dict, evals_dict: typing.Dict) -> typing.List[typing.Tuple[int, float]]:
        self.method = 'LCCV'
        if best_performance is None:
            config["anchor_size"] = self.final_anchor
            evals_dict[self.final_anchor] += 1
            result = self.surrogate_model.predict(pd.DataFrame([dict(config)]))[0]
            return [(self.final_anchor, result)], evals_dict
        
        results = []
        for anchor in self.anchors:
            evals_dict[anchor] += 1
            config["anchor_size"] = anchor
            performance = self.surrogate_model.predict(pd.DataFrame([dict(config)]))[0]
            results.append((anchor, performance))

            if len(results) >= 2 and self.optimistic_extrapolation(
                results[-2][0], results[-2][1], results[-1][0], results[-1][1], self.final_anchor
            ) > best_performance:
                break     
        return results, evals_dict
