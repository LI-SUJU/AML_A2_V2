from abc import ABC
from sklearn.pipeline import Pipeline
import typing


class VerticalModelEvaluator(ABC):
    '''
    VerticalModelEvaluator is a class that evaluates a surrogate model's performance over a range of anchor sizes.
    Attributes:
        surrogate_model (Pipeline): A sklearn pipeline object that has been fitted on LCDB data. It is used to predict 
                                    the performance of a configuration given a numpy array consisting of configuration 
                                    information and an anchor size.
        anchors (list): A list of anchor sizes to be used for evaluation.
        minimal_anchor: The smallest anchor size to be used.
        final_anchor: The largest anchor size to be used.
    Methods:
        __init__(surrogate_model: Pipeline, anchors: list) -> None:
            Initializes the VerticalModelEvaluator with a surrogate model and a list of anchors.
            Parameters:
                surrogate_model (Pipeline): A sklearn pipeline object that has already been fitted on LCDB data.
                anchors (list): A list of anchor sizes to be used for evaluation.
        evaluate_model(best_so_far: float, configuration: typing.Dict) -> typing.List[float]:
            Evaluates the model's performance over the range of anchor sizes and returns the learning curve.
            Parameters:
                best_so_far (float): The best performance observed so far.
                configuration (typing.Dict): A dictionary containing the configuration information.
            Returns:
                typing.List[float]: A list of tuples where each tuple contains an anchor size and the corresponding 
                                    expected performance.
    '''

    def __init__(self, surrogate_model: Pipeline, anchors: list) -> None:
        self.surrogate_model = surrogate_model
        self.anchors = anchors
        self.minimal_anchor = anchors[0]
        self.final_anchor = anchors[-1]

    def evaluate_model(self, best_so_far: float, configuration: typing.Dict) -> typing.List[float]:
        anchor = self.minimal_anchor
        learning_curve = []
        for anchor in self.anchors:
            configuration['anchor_size'] = anchor 
            expected_performance = self.pipeline(configuration)
            learning_curve.append((anchor,expected_performance))

        return learning_curve

