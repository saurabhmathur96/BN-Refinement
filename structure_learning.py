import random
from math import log
from os import path
from typing import *
from itertools import product

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import BIFReader

bn_path = "bns"
BN_NAMES =  ["asia", "alarm", "sachs", "child"]

def get_bn(name: str):
    if name not in BN_NAMES: 
      raise ValueError(f'Unknown BN "{name}"')
    return BIFReader(path.join(bn_path, f"{name}.bif")).get_model()


def sample_from_bn(M: BayesianNetwork, n: int):
    return M.simulate(n_samples=n)


def sample_edges(M: BayesianNetwork, n_edges: int):
    V = list(M.nodes)
    E = list(M.edges)

    E0 = random.sample(E, n_edges)
    M0 = BayesianNetwork()
    M0.add_nodes_from(V)
    M0.add_edges_from(E0)
    return M0


def refine_bn(
    M0: BayesianNetwork, data: pd.DataFrame, state_names: Dict[str, List], **kwargs
):
    """
    Parameters:
      M0: Initial BN
      data: Dataframe containing data set
      state_names: dict of variable name -> list of values the variable can take

    Returns:
      G: The refined BN

    """
    estimator = HillClimbSearch(data, state_names=state_names)
    if "scoring_method" not in kwargs:
        kwargs["scoring_method"] = "bicscore"
        
    M = estimator.estimate(start_dag=M0, **kwargs)
    return M


def SHD(optimal, estimated):
    """
    Source: https://github.com/mj-sam/pgmpy-upgraded
    Parameter :
        optimal :
            the optimal learned graph object
        estimated :
            the estimated learned graph object
    """
    opt_edges = set(optimal.edges())
    est_edges = set(estimated.edges())
    opt_not_est = opt_edges.difference(est_edges)
    est_not_opt = est_edges.difference(opt_edges)
    c = 0
    for p1 in opt_not_est:
        for p2 in est_not_opt:
            if set(p1) == set(p2):
                c += 1
    SHD_score = len(opt_not_est) + len(est_not_opt) - c

    """
    References
    ---------
    [1] de Jongh, M. and Druzdzel, M.J., 2009. A comparison of structural distance
        measures for causal Bayesian network models. Recent Advances in Intelligent
        Information Systems,Challenging Problems of Science, Computer Science series,
        pp.443-456.
    """
    return SHD_score


def compute_tree_bic(
    dt: DecisionTreeClassifier, X: np.ndarray, y: np.ndarray, states: List
):
    """Computes the BIC for a tree-CPD"""

    n_nodes = dt.tree_.node_count
    children_left = dt.tree_.children_left
    children_right = dt.tree_.children_right

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    n_leaves = dt.get_n_leaves()

    sample_size = len(y)
    y_pred = dt.predict_proba(X)
    log_likelihood = -log_loss(
        y, y_pred, normalize=False, labels=dt.classes_
    )
    penalty = (
        -n_nodes
        - np.sum(np.log(n_nodes - node_depth[is_leaves == 0]))
        - 0.5 * log(sample_size) * (n_leaves * (len(states) - 1))
    )
    return log_likelihood + penalty


def convert_to_table_cpd(dt, variable: str, parents: List[str], state_names: Dict, sample_size:int):
    parents_card = [len(state_names[parent]) for parent in parents]
    values = np.zeros((np.prod(parents_card), len(state_names[variable])))
    # values /= values.sum(axis=0, keepdims=True)

    X = pd.DataFrame(list(product(*[state_names[parent] for parent in parents])), columns=parents)
    p = dt.predict_proba(X)
    for i, pi in enumerate(p): 
      prob = { c: pij for c, pij in zip(dt.classes_, pi) }

      values[i] = [prob[state] for state in state_names[variable]]
    cpd = TabularCPD(variable, len(state_names[variable]), values.T, parents, parents_card, state_names=state_names)
    
    return cpd



class StructuredBicScore(BicScore):
    """
    BIC with tree-structured CPDs
    """

    def __init__(self, data, **kwargs):
        super(StructuredBicScore, self).__init__(data, **kwargs)

    def local_score(self, variable, parents, return_cpd=False):
        """Computes BIC such that CPDs can possibly be tree-structured"""

        table_bic = super().local_score(variable, parents)

        model = BayesianNetwork()
        model.add_nodes_from(parents + [variable])
        model.add_edges_from([(parent, variable) for parent in parents])
        table_cpd = BayesianEstimator(model, self.data, state_names=self.state_names).estimate_cpd(variable, prior_type="dirichlet", pseudo_counts=1) 
        
    
        if len(parents) == 0 or self.data[variable].nunique() < len(self.state_names[variable]):
            if return_cpd:
                return table_bic, table_cpd
            return table_bic

        var_states = self.state_names[variable]
        
        variables = parents+[variable]
        df = self.data[variables]
        
        data_encoder = OneHotEncoder(
            categories=[self.state_names[each] for each in parents], drop="if_binary"
        )
        X = data_encoder.fit_transform(df[parents])
        y = df[variable]
        dt = DecisionTreeClassifier(criterion="log_loss")

        scores = [table_bic]
        max_leaves = np.power(2, np.arange(2, X.shape[1])) 
        
        for threshold in max_leaves:
            dt.set_params(max_leaf_nodes=threshold)
            dt.fit(X, y)

            scores.append(compute_tree_bic(dt, X, y, var_states))
        
    
        if return_cpd:
            i = np.argmax(scores)
            if i == 0:
                return np.max(scores), table_cpd
            else:
                dt = DecisionTreeClassifier(criterion="log_loss")
                dt.set_params(max_leaf_nodes=max_leaves[i-1])
                
                dt.fit(X, y)

                for node_id in range(dt.tree_.node_count):
                    dt.tree_.value[node_id] += 1

                cpd =  convert_to_table_cpd(Pipeline([("encoder", data_encoder), ("tree", dt)]), 
                                            variable, parents, self.state_names, len(y))
                return np.max(scores), cpd

        return np.max(scores)  # >= Table-BIC
