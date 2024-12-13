import ConfigSpace
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import pandas as pd
from tools import preprocess_configurations

class SurrogateModel:
    """Train a random forest regressor on hyperparameter configurations and their scores."""
    
    def __init__(self, config_space: str):
        self.config_space = config_space
        self.model = Pipeline([('model', RandomForestRegressor())])
        self.features = None

    def fit(self, df: pd.DataFrame):
        """Fit the model on the provided dataframe."""
        y = df.iloc[:, -1]
        self.features = df.columns[:-1]
        df_preprocessed = preprocess_configurations(self.config_space, df)
        self.model.fit(df_preprocessed[self.features], y)

    def predict(self, theta_new):
        """Predict the performance of a given configuration."""
        if isinstance(theta_new, list):
            X = pd.DataFrame(theta_new)
        elif isinstance(theta_new, ConfigSpace.Configuration):
            X = pd.DataFrame([dict(theta_new)])
        else:
            X = pd.DataFrame([theta_new])
        
        for col in self.features:
            if col not in X.columns:
                X[col] = None
        
        X_preprocessed = preprocess_configurations(self.config_space, X)
        return self.model.predict(X_preprocessed[self.features])
