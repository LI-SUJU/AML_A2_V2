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
    Encodes the configurations from a dataframe based on the given ConfigSpace.

    Args:
        config_space (ConfigSpace.ConfigurationSpace): The configuration space defining hyperparameters.
        df (pd.DataFrame): DataFrame containing configurations to be encoded.

    Returns:
        pd.DataFrame: Transformed DataFrame with missing values filled and categorical values mapped to integers.
    """
    # Create mappings for categorical hyperparameters
    categories = {}
    for param in config_space.get_hyperparameters():
        if isinstance(param, ConfigSpace.hyperparameters.CategoricalHyperparameter):
            categories[param.name] = {choice: num for num, choice in enumerate(param.choices)}

    # Create a copy of the DataFrame
    df = df.copy()

    # Fill missing values with default values from the configuration space
    for param in config_space.get_hyperparameters():
        if param.name not in df.columns:
            df[param.name] = param.default_value
        else:
            df[param.name] = df[param.name].fillna(param.default_value).infer_objects(copy=False)


    # Map categorical columns to numerical values
    for column in df.columns:
        if column in categories:
            df[column] = df[column].map(categories[column])

    # Reorder columns to match the order in the configuration space
    original_ordering = [param.name for param in config_space.get_hyperparameters()]
    if "anchor_size" in df.columns:
        original_ordering.append("anchor_size")
    df = df[original_ordering]

    return df
