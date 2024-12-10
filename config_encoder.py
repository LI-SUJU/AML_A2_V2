import ConfigSpace
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

'''
The class implements the instance methods fit() and transform(). 
-> need to have both X and y parameters, and transform() should return a pandas DataFrame

map category values to numbers from configspace
self can be self
transform should map categories to numbers

'''
class ConfigEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, config_space):
        self.config_space = config_space
        self.categories = self._mappings()
        
    def _mappings(self):
        mappings = {}
        
        for param in self.config_space.get_hyperparameters():
            if isinstance(param, ConfigSpace.hyperparameters.CategoricalHyperparameter):
                mappings[param.name] = {choice: num for num, choice in enumerate(param.choices)}
                
        return mappings
    
    def fit(self, X, y=None):
        return self            

    # imputer sets unspecified parameters to median, but want this to be default param from configspace
    def transform(self, df):
        df = df.copy()
        
        for param in self.config_space.values():
            # either the column doesnt exist -> add and fill with default
            if param.name not in df.columns:
                df[param.name] = param.default_value
            # or it does, but there are missing values -> fill with defaults
            else:
                df[param.name] = df[param.name].fillna(param.default_value)
        for column in df.columns:
            if column in self.categories:
                df[column] = df[column].map(self.categories[column])        

        # fix error with order of params
        original_ordering = [param.name for param in self.config_space.get_hyperparameters()]
        # anchor size not a parameter in json, so append
        original_ordering.append("anchor_size")
        df = df[original_ordering]
        
        return df