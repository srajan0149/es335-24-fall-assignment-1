"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)


@dataclass

class Node():
    def __init__(self,feature = None,threshold = None,left = None,right = None,info_gain = None,value=None):

        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value

    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None

    def build(self, X: pd.DataFrame, y: pd.Series, depth = 0):
        if len(y)==0:
            return None
        num_samples, num_features = np.shape(X)
        features = pd.Series(X.columns)
        # splitting till max depth
        if depth<self.max_depth:
            
            # finding the best split
            best_attribute = opt_split_attribute(X, y, self.criterion, features)

            threshold_value = best_split_point(y,X[best_attribute])[0]

            # splitting the dataset
            split_dataset = split_data(X, y, X[best_attribute], threshold_value)

            # recur left
            left_subtree = self.build(split_dataset[0], split_dataset[2], depth+1)
            # recur right
            right_subtree = self.build(split_dataset[1], split_dataset[3], depth+1)
            # return decision node
            return Node(best_attribute, threshold_value, left_subtree, right_subtree, information_gain(y,X[best_attribute],criterion=self.criterion))
        
        # leaf_value = self.calculate_leaf_value(X.iloc[:, -1])
        leaf_value = self.calc_leaf_value(y)
        return Node(value=leaf_value)    

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 

        self.tree = self.build(X,y)
        pass

    def _predict_single_row(self, x: pd.Series, curr_node):
        if curr_node is None:
            return 0
        
        if curr_node.is_leaf():
                return curr_node.value
            
        feature_value = x.iloc[curr_node.feature]
            
        if feature_value<curr_node.threshold:
            return self._predict_single_row(x, curr_node.left)
        else:
            return self._predict_single_row(x, curr_node.right)


    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.
        if not check_ifreal(X.iloc[:,0]):
            X = one_hot_encoding(X)

        predictions = pd.Series([self._predict_single_row(S,self.tree) for x,S in X.iterrows()])
        return predictions
    
    def _print_tree_recursive(self, node, depth):
        if node is None:
            return

        indent = '    ' * depth
        if node.is_leaf():
            print(f"{'Value:'} {node.value}")
        else:
            print(f"?(feature'{node.feature}' > {node.threshold})")
            print(f"{indent}   Y: ", end="")
            self._print_tree_recursive(node.left, depth + 1)
            print(f"{indent}   N: ", end="")
            self._print_tree_recursive(node.right, depth + 1)

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        self._print_tree_recursive(self.tree, depth=0)

    def calc_leaf_value(self, Y):
        ''' function to compute leaf node '''
        if len(Y)==0:
            return None
        Y = pd.Series(Y)
        if(check_ifreal(Y)):
            return np.mean(Y)
        else:
            counts = Y.value_counts()
            most_occuring_value = counts.idxmax()
            return most_occuring_value