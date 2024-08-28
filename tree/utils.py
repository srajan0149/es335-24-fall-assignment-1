"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""
import numpy as np
import pandas as pd

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data.
    """

    X = pd.get_dummies(X , columns = X.columns)
    return X

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """

    if len(y.unique()) <= len(y)**0.5:
        return False
    else:
        return True


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    Note: Only applicable for discrete input data.
    """

    count_values = Y.value_counts()
    probab_value = count_values / len(Y)
    entropy = -np.sum(probab_value * np.log2(probab_value))
    return entropy


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index.
    Note: Only applicable for discrete input data.
    """

    count_values = Y.value_counts()
    probab_value = count_values / len(Y)
    gini = -np.sum(probab_value**2)
    return gini + 1



def MSE(Y: pd.Series) -> float:
    """
    Function to calculate mean square error aka variance.
    Note: Only applicable for real input data
    """
    if len(Y)==0:
        return 0
    Y_mean = np.mean(Y)
    Y2 = (Y - Y_mean)**2 
    mean_sqrd_err = np.mean(Y)

    return mean_sqrd_err

def best_split_point(Y: pd.Series, column: pd.Series) -> list:
    # Criterion has to be MSE

    if len(column) == 1:
        return [column.iloc[0],0]
    elif len(column) == 0:
        return 
    
    S = column.sort_values()

    best_split_value = (S.iloc[0]+S.iloc[1])/2
    Y_left = S[S<=best_split_value]
    Y_right = S[S>best_split_value]
    min_mse = mse = (len(Y_left)*MSE(Y_left) + len(Y_right)*MSE(Y_right))/len(Y)

    for i in range(len(S)-1):
        split_value = (S.iloc[i]+S.iloc[i+1])/2
        Y_left = S[S<=best_split_value]
        Y_right = S[S>best_split_value]
        mse = (len(Y_left)*MSE(Y_left) + len(Y_right)*MSE(Y_right))/len(Y)
        if mse < min_mse:
            min_mse = mse
            best_split_value = split_value

    return [best_split_value,min_mse]


def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini_index or MSE)
    """

    if (criterion == "entropy"):
        impurity_before = entropy(Y)
    elif criterion == "gini_index":
        impurity_before = gini_index(Y)
    else:
        impurity_before = MSE(Y)

    impurity_after = 0

    if check_ifreal(attr):

        impurity_after = best_split_point(Y,attr)[1]

    else:

        for label in attr.unique():
            values_given_label = Y[attr == label]
            if criterion == 'entropy':
                sub_impurity_after = entropy(values_given_label)
            elif criterion == 'gini_index':
                sub_impurity_after = gini_index(values_given_label)
            else:
                sub_impurity_after = MSE(values_given_label)
        
            impurity_after += (len(values_given_label)/Y.size) * sub_impurity_after
    
    information_gain = impurity_before - impurity_after

    return information_gain


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).

    max_info_gain = 0
    opt_attrbt = None

    for _ in features:
        info_gain = information_gain(y, X[_], criterion)

        if info_gain >= max_info_gain:
            max_info_gain = info_gain
            opt_attrbt = _
        
    return opt_attrbt


def split_data(X: pd.DataFrame, y: pd.Series, column, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.

    if check_ifreal(column):
        left = (column <= value)
        right = (column >= value)
    else:
        left = (column == value)
        right = (column != value)

    X_left, X_right = X[left], X[right]
    y_left, y_right = y[left], y[right]

    return X_left, X_right, y_left, y_right
