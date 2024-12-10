import typing
import pandas as pd
from vertical_model_evaluator import VerticalModelEvaluator

class LCCV(VerticalModelEvaluator):
    
    @staticmethod
    def optimistic_extrapolation(
        previous_anchor: int, previous_performance: float, 
        current_anchor: int, current_performance: float, target_anchor: int
    ) -> float:
        """
        Does the optimistic performance. Since we are working with a simplified
        surrogate model, we can not measure the infimum and supremum of the
        distribution. Just calculate the slope between the points, and
        extrapolate this.

        :param previous_anchor: See name
        :param previous_performance: Performance at previous anchor
        :param current_anchor: See name
        :param current_performance: Performance at current anchor
        :param target_anchor: the anchor at which we want to have the
        optimistic extrapolation
        :return: The optimistic extrapolation of the performance
        """
        slope = (previous_performance - current_performance) / (previous_anchor - current_anchor )
        extrapolated = current_performance + (target_anchor - current_anchor) * slope
        return extrapolated
    
    def evaluate_model(self, best_so_far: typing.Optional[float], configuration: typing.Dict, evaluations_dict: typing.Dict) -> typing.List[typing.Tuple[int, float]]:
        """
        Does a staged evaluation of the model, on increasing anchor sizes.
        Determines after the evaluation at every anchor an optimistic
        extrapolation. In case the optimistic extrapolation can not improve
        over the best so far, it stops the evaluation.
        In case the best so far is not determined (None), it evaluates
        immediately on the final anchor (determined by self.final_anchor)

        :param best_so_far: indicates which performance has been obtained so far
        :param configuration: A dictionary indicating the configuration

        :return: A tuple of the evaluations that have been done. Each element of
        the tuple consists of two elements: the anchor size and the estimated
        performance.
        """
        self.method = 'LCCV'
        if best_so_far == None:
            
            configuration["anchor_size"] = self.final_anchor
            evaluations_dict[self.final_anchor]+=1
            config = pd.DataFrame([dict(configuration)])
            result = self.surrogate_model.predict(config)[0]
            return ([(self.final_anchor, result)], evaluations_dict)
        results = []

        for anchor in self.anchors:
            evaluations_dict[anchor]+=1
            configuration["anchor_size"] = anchor
            config = pd.DataFrame([dict(configuration)])
            performance = self.surrogate_model.predict(config)[0]
            results.append((anchor, performance))

            if len(results) >= 2: 
                extrapolated = self.optimistic_extrapolation(results[-2][0], results[-2][1], #  previous_anchor: int, previous_performance: float, 
                                                             results[-1][0], results[-1][1], # current_anchor: int, current_performance: float,
                                                             self.final_anchor)
                # print (f"extrapolated {extrapolated}, best_so_far { best_so_far}")
                if extrapolated > best_so_far:
                    break     
        return (results, evaluations_dict)

        
            
