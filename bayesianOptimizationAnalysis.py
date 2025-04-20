import ray
from ray import tune
from ray.tune import ExperimentAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
analysis = ExperimentAnalysis(r"C:\Users\Tristan\Downloads\HyPCAR3\ray_results\ray_tune_multi_objective")
df = analysis.dataframe()

# Optionally, inspect the first few rows
# print(df.head())
# print(type(df))
print(analysis.get_best_config(metric="mse",mode="min"))
print(analysis.get_best_config(metric="kl",mode="min"))
print(analysis.get_best_config(metric="topk",mode="max"))
print(analysis.get_best_config(metric="ce",mode="min"))
# plt.figure(figsize=(10, 6))

