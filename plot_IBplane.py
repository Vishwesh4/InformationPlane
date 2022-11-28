import os
from pathlib import Path

import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


FILE_NAME = "./Results/mi_tanh_test.npy"

with open(FILE_NAME,"rb") as file:
    XH = np.load(file)
    YH = np.load(file)

n_epochs, n_layers = XH.shape

colors = cm.rainbow(np.linspace(0, 1, n_epochs))

fig = plt.figure()
markers = ["x","o","^","v"]
labels = ["First layer", "Second layer", "Third layer", "Fourth layer"]

for epoch in range(n_epochs):
    for layer in range(n_layers):
        plt.scatter(XH[epoch,layer],YH[epoch,layer],marker=markers[layer],color=colors[epoch])
    # plt.plot(XH[epoch,:],YH[epoch,:],"k--")
points = [Line2D([0],[0],marker=markers[layer], color='k', label=labels[layer], markersize=10, markerfacecolor="k") for layer in range(n_layers)]
sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap('rainbow'))
cbar = plt.colorbar(sm, ticks=[0,1])
cbar.ax.set_yticklabels(['0',str(10*n_epochs)])
plt.legend(handles=points)
# plt.show()
plt.savefig(str(Path(FILE_NAME).parent/Path(FILE_NAME).stem)+".png")