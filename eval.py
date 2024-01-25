from os import path
import pandas as pd
from pgmpy.readwrite import BIFReader, UAIReader
from pgmpy.metrics import log_likelihood_score

bn_path = "results"
bn_name = "garbh-ini-1"
M = BIFReader(path.join(bn_path, f"{bn_name}.bif")).get_model()

data_name = "numom2b"
data = pd.read_csv(path.join("data", f"{data_name}.csv")).astype(str)
if "Parity" in M.nodes and "Parity" not in data.columns:
    data["Parity"] = "0"

if "Parity" not in M.nodes and "Parity" in data.columns:
    data = data.drop("Parity", axis=1)
# BIFReader reads state names as str by default. So, converting data to str

print (f"{bn_name},{data_name},{log_likelihood_score(M, data)}")
