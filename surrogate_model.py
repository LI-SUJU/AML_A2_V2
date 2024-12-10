import ConfigSpace

import sklearn.impute
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import pandas as pd
from config_encoder import ConfigEncoder
import numpy as np 
import matplotlib.pyplot as plt 

class SurrogateModel:
    """ Class to train a random forest regressor on a dataframe with hyperparameter configurations and 
    their resulting score as the last column. Once fitted the regressor can receive configurations in either
    a list (for multiple configuration), as a ConfigSpace configuration or as a dictionary for predictions.
    """    
    def __init__(self, config_space:str):
        self.config_space = config_space
        self.df = None
        self.encoder = ConfigEncoder(self.config_space)
        self.model = Pipeline([
                               ('model', RandomForestRegressor())])  

    def fit(self, df:pd.DataFrame):
        """
        Receives a data frame, in which each column (except for the last two) represents a hyperparameter, the
        penultimate column represents the anchor size, and the final column represents the performance.

        :param df: the dataframe with performances
        :return: Does not return anything, but stores the trained model in self.model
        """
        y = df['score']
        self.features = df.columns[:-1]
        self.label = df.columns[-1]
        df_encoded = self.encoder.transform(df)
        self.df = df_encoded
        self.model.fit(df_encoded[self.features],y)

    def predict(self, theta_new):
        """
        Predicts the performance of a given configuration theta_new

        :param theta_new: a dict, where each key represents the hyperparameter (or anchor)
        :return: float, the predicted performance of theta new (which can be considered the ground truth)
        """
        if isinstance(theta_new, list): # len(theta_new)>1:
            X = pd.DataFrame(theta_new,index=[x for x in range(len(theta_new))])
        elif isinstance(theta_new, ConfigSpace.Configuration):
            X = pd.DataFrame([dict(theta_new)])
        else:
            X = pd.DataFrame(theta_new)
        for col in self.features:
            if col not in X.columns:
                X[col] = None
        T = self.encoder.transform(X)
        # print(T)
        return self.model.predict(T[self.features])


data_sets = ['config_performances_dataset-6.csv',
             'config_performances_dataset-11.csv',
             'config_performances_dataset-1457.csv']

def main():
    data_sets = ['config_performances_dataset-6.csv',
                'config_performances_dataset-11.csv',
                'config_performances_dataset-1457.csv']
    colors = ['tab:blue','tab:green','tab:red']
    config_space = ConfigSpace.ConfigurationSpace.from_json('lcdb_config_space_knn.json')
    results = {}
    for data_set,color in zip(data_sets,colors):
        name = data_set.split('_')[-1][:-4]
        print(name, 'started')
        data = pd.read_csv(data_set)
        plt.hist(data['score'], bins=30, color=color, alpha=0.3, label=name)
        errors = []
        r2s = []
        for i in range(10):
            train_data, test_data = train_test_split(data)
            model = SurrogateModel(config_space)
            model.fit(train_data)
            y_pred = model.predict(test_data)
            errors.append( mean_squared_error(y_pred,test_data['score']))
            r2s.append(r2_score(y_pred,test_data['score']))
        
        results[name] = {'error':f'{np.mean(errors):.3e},std {np.std(errors):.2e}',
                         'r2':f'{np.mean(r2s):.2f},std {np.std(r2s):.2f}'}
        pd.DataFrame(results).to_csv('surrogate_model_verified.csv')
    print(results)
    plt.legend()
    plt.title('Distribution of scores for datasets')
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('surrogate_data.png')  
        

if __name__ == '__main__':
    main()

