import numpy as np
import ConfigSpace

def calculate_evaluations(evaluation_dict, weight_function):
    return sum(weight_function(anchor) * evaluations for anchor, evaluations in evaluation_dict.items())

def calculate_linear_evaluations(evaluation_dict):
    return calculate_evaluations(evaluation_dict, lambda x: x)

def calculate_log_evaluations(evaluation_dict):
    return calculate_evaluations(evaluation_dict, np.log)

def calculate_square_evaluations(evaluation_dict):
    return calculate_evaluations(evaluation_dict, lambda x: x**2)

def preprocess_configurations(config_space, df):

    """
    Preprocess a DataFrame with hyperparameter configurations to match the order of the configuration space.
    This function fills missing values with default values and maps categorical columns to numerical values.
    """

    # Create mappings for categorical hyperparameters
    categories = {param.name: {choice: num for num, choice in enumerate(param.choices)}
                  for param in config_space.get_hyperparameters()
                  if isinstance(param, ConfigSpace.hyperparameters.CategoricalHyperparameter)}

    # Fill missing values with defaults
    df = df.copy()
    for param in config_space.get_hyperparameters():
        df[param.name] = df.get(param.name, param.default_value).fillna(param.default_value).infer_objects(copy=False)

    # Map categorical columns to numerical values
    df = df.apply(lambda col: col.map(categories[col.name]) if col.name in categories else col)

    # Reorder columns to match the order in the configuration space
    original_ordering = [param.name for param in config_space.get_hyperparameters()]
    if "anchor_size" in df.columns:
        original_ordering.append("anchor_size")
    return df[original_ordering]
