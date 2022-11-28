# Information Plane Experiments for Neural Networks, Graph Neural Networks and Deep Graph Infomax
In this project, I have explored Information plane for different types of architecture exploring different modalities of data such as image and graphs. In particular, I was interested in how the information plane would look like for Infomax.

## Getting Started
### Prerequisites 

Python packages
```
numpy
tqdm
torch
torchvision
matplotlib
torch-sparse
torch-scatter
torchmetrics
```

### How to run
1. To construct information plane for NN and GNN, run the files `mnist_relu.py`,`mnist_tanh.py` or `gnn_ib.py`
2. I found EDGE[link](https://github.com/mrtnoshad/EDGE/tree/master/information_plane) to work best for calculation of mutual information
3. Construct the plot based on the `.npy` file generated in `Results` folder, using `plot_IBplane.py`
4. For constructing information plane for Deep Graph Infomax[link](https://github.com/fanyun-sun/InfoGraph/tree/master/unsupervised), run the command
```
$ python main.py --DS DD --lr 0.001 --num-gc-layers 3
```
and construct the plots using `plot_IBplane.py`