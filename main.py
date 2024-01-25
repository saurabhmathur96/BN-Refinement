import pandas as pd

from structure_learning import get_bn, sample_from_bn, sample_edges
from structure_learning import refine_bn, StructuredBicScore
from structure_learning import SHD, BN_NAMES
from pgmpy.metrics import log_likelihood_score
from pgmpy.estimators import BayesianEstimator


rows = []
for name in BN_NAMES:
    M = get_bn(name)
    print (f"Loaded {name}")
    for _ in range(5):
        train = sample_from_bn(M, 1_000)
        test = sample_from_bn(M, 1_000)
        M0 = sample_edges(M, int(0.5*len(M.edges)))

        M1 = refine_bn(M0.copy(), train, M.states, scoring_method="bicscore", tabu_length=1)
        M1.fit(train, state_names=M.states, estimator=BayesianEstimator,  prior_type="dirichlet", pseudo_counts=1)
        shd0 = SHD(M, M1)
        ll0 = log_likelihood_score(M1, test)

        scorer = StructuredBicScore(train, state_names=M.states)
        M2 = refine_bn(M0.copy(), train, M.states, scoring_method=scorer, tabu_length=1)
        cpds = []
        for node in M2.nodes:
            parents = M2.get_parents(node)
            _, cpd = scorer.local_score(node, parents, return_cpd=True)
            cpds.append(cpd)
        M2.add_cpds(*cpds)
        shd1 = SHD(M, M2)
        ll1 = log_likelihood_score(M2, test)

        rows.append([name, shd0, shd1, ll0, ll1])

print (pd.DataFrame(rows, columns=["Data set", "SHD (BIC)", "SHD (S-BIC)", "LL (BIC)", "LL (S-BIC)"]).groupby(by="Data set").mean())

