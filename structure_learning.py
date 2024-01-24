from pgmpy.estimators import StructureScore, BicScore
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
from math import log

from sklearn import tree
from os import path

bn_path = ""
def get_bn(name: str):
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


def refine_bn(M0: BayesianNetwork, data: pd.DataFrame, state_names: Dict[str, List], **kwargs):
  """
    Parameters:
      M0: Initial BN
      data: Dataframe containing data set
      state_names: dict of variable name -> list of values the variable can take

    Returns:
      G: The refined BN

  """
  estimator = HillClimbSearch(data, state_names=state_names)
  M = estimator.estimate(scoring_method=scoring_method, start_dag=M0, **kwargs)
  return M



def SHD(optimal,estimated):
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
    c = 0;
    for p1 in opt_not_est:
        for p2 in est_not_opt:
            if(set(p1) == set(p2)):
                c +=1
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



def compute_tree_bic(dt: DecisionTreeClassifier, X: np.ndarray, y: np.ndarray, states: List):
  """ Computes the BIC for a tree-CPD """
  
  n_nodes = dt.tree_.node_count
  children_left = dt.tree_.children_left
  children_right = dt.tree_.children_right
  feature = dt.tree_.feature
  threshold = dt.tree_.threshold
  values = dt.tree_.value

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
  n_internal = n_nodes - n_leaves

  y_pred = dt.predict_proba(X)
  log_likelihood = -log_loss(y, y_pred, normalize=False, labels=list(range(len(self.state_names[variable]))))
  penalty = -n_nodes - np.sum(np.log(n_nodes - node_depth[is_leaves == 0])) - 0.5*log(sample_size) * (n_leaves * (len(states)-1))
  return log_likelihood + penalty


class StructuredBicScore(BicScore):
    """
    BIC with tree-structured CPDs
    """

    def __init__(self, data, **kwargs):
        super(StructuredBicScore, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        """ Computes BIC such that CPDs can possibly be tree-structured """

        table_bic = super().local_score(variable, parents)
        if len(parents) == 0 or self.data[variable].nunique() == 1: 
          return table_bic
        
        var_states = self.state_names[variable]
        sample_size = len(self.data)
        
        X = OneHotEncoder(
            categories = [self.state_names[each] for each in parents],
            drop="if_binary"
        ).fit_transform(self.data[parents])
        y = LabelEncoder().fit_transform(self.data[variable])
        dt = DecisionTreeClassifier(criterion = "log_loss")
        path = dt.cost_complexity_pruning_path(X, y)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        ccp_alphas = np.clip(ccp_alphas, 0, np.inf)
        
        scores = [base]
        for ccp_alpha in ccp_alphas:
          dt.set_params(ccp_alpha = ccp_alpha)
          dt.fit(X, y)
          scores.append(compute_bic(dt, X, y, var_states))
        return np.max(scores) # >= Table-BIC
        
