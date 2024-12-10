import typing
import pandas as pd
from vertical_model_evaluator import VerticalModelEvaluator

class LCCV(VerticalModelEvaluator):
    
    @staticmethod
    def optimistic_extrapolation(
        prev_anchor: int, prev_performance: float, 
        curr_anchor: int, curr_performance: float, target_anchor: int
    ) -> float:
        """
        Performs optimistic extrapolation. Since we are working with a simplified
        surrogate model, we cannot measure the infimum and supremum of the
        distribution. Just calculate the slope between the points, and
        extrapolate this.

        :param prev_anchor: Previous anchor point
        :param prev_performance: Performance at previous anchor
        :param curr_anchor: Current anchor point
        :param curr_performance: Performance at current anchor
        :param target_anchor: The anchor at which we want to have the
        optimistic extrapolation
        :return: The optimistic extrapolation of the performance
        """
        slope = (prev_performance - curr_performance) / (prev_anchor - curr_anchor)
        extrapolated_performance = curr_performance + (target_anchor - curr_anchor) * slope
        return extrapolated_performance
    
    def evaluate_model(self, best_performance: typing.Optional[float], config: typing.Dict, evals_dict: typing.Dict) -> typing.List[typing.Tuple[int, float]]:
        """
        Performs a staged evaluation of the model on increasing anchor sizes.
        Determines after the evaluation at every anchor an optimistic
        extrapolation. If the optimistic extrapolation cannot improve
        over the best performance so far, it stops the evaluation.
        If the best performance so far is not determined (None), it evaluates
        immediately on the final anchor (determined by self.final_anchor).

        :param best_performance: Indicates the best performance obtained so far
        :param config: A dictionary indicating the configuration
        :param evals_dict: A dictionary to keep track of evaluations

        :return: A tuple of the evaluations that have been done. Each element of
        the tuple consists of two elements: the anchor size and the estimated
        performance.
        """
        self.method = 'LCCV'
        if best_performance is None:
            config["anchor_size"] = self.final_anchor
            evals_dict[self.final_anchor] += 1
            config_df = pd.DataFrame([dict(config)])
            result = self.surrogate_model.predict(config_df)[0]
            return ([(self.final_anchor, result)], evals_dict)
        
        results = []

        for anchor in self.anchors:
            evals_dict[anchor] += 1
            config["anchor_size"] = anchor
            config_df = pd.DataFrame([dict(config)])
            performance = self.surrogate_model.predict(config_df)[0]
            results.append((anchor, performance))

            if len(results) >= 2: 
                extrapolated_performance = self.optimistic_extrapolation(
                    results[-2][0], results[-2][1],  # prev_anchor, prev_performance
                    results[-1][0], results[-1][1],  # curr_anchor, curr_performance
                    self.final_anchor
                )
                if extrapolated_performance > best_performance:
                    break     
        return (results, evals_dict)
