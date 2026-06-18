import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata
from ray import tune
from ray.tune import ExperimentAnalysis
# --- assume df is your DataFrame already loaded ---
analysis=ExperimentAnalysis(r"C:\Users\Tristan\Downloads\HyPCAR3\ray_results\ray_tune_multi_objective")
df=analysis.dataframe()
prettify={"mse":"Mean-Squared-Error","topk":"Top-1 Accuracy","ce":"Cross-Entropy","kl":"KL-Divergence"}

import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata

for metric in ["mse","topk","ce","kl"]:
    x_param="config/lr"
    y_param="config/Drop1"

    #Create grid points
    x_vals=np.linspace(df[x_param].min(),df[x_param].max(), 50)
    y_vals=np.linspace(df[y_param].min(),df[y_param].max(), 50)
    X,Y=np.meshgrid(x_vals, y_vals)

    #Interpolate the metric values on the grid
    points=df[[x_param, y_param]].values
    values=df[metric].values
    Z=griddata(points,values,(X, Y),method='linear')
    # print(Z)

    fig=plt.figure(figsize=(10, 8))
    ax=fig.add_subplot(111, projection='3d')
    surf=ax.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor='none', alpha=0.8)
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_zlabel(prettify[metric])
    ax.set_title("3D Surface Plot of {} over Dropout-2 and Dropout-1".format(prettify[metric]))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    filePath="3D_surface_"+metric+".png"
    plt.savefig(filePath)
    plt.show()