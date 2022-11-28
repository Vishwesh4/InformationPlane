# Information Plane Experiments for Neural Networks, Graph Neural Networks and Deep Graph Infomax
In this course project, I have explored Information plane for different types of architecture exploring different modalities of data such as image and graphs. In particular, I was interested in how the information plane would look like for Infomax.

## Getting Started

### Prerequisites 

Python packages required
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
2. I found EDGE [(link)](https://github.com/mrtnoshad/EDGE/tree/master/information_plane) to work best for calculation of mutual information
3. Construct the plot based on the `.npy` file generated in `Results` folder, using `plot_IBplane.py`
4. For constructing information plane for Deep Graph Infomax  [(link)](https://github.com/fanyun-sun/InfoGraph/tree/master/unsupervised), run the command
```
$ python main.py --DS DD --lr 0.001 --num-gc-layers 3
```
and construct the plots using `plot_IBplane.py`

## Results
<img src="https://github.com/Vishwesh4/MAT1510-CourseProject/blob/master/Results/mi_tanh_test.png" align="center" width="500"><figcaption>Fig.1 - Information Plane for Neural Network(tanh activation) trained on MNIST</figcaption></a>  
  
<img src="https://github.com/Vishwesh4/MAT1510-CourseProject/blob/master/Results/mi_relu_test.png" align="center" width="500"><figcaption>Fig.2 -  Information Plane for Neural Network(ReLu activation) trained on MNIST</figcaption></a>  

<img src="https://github.com/Vishwesh4/MAT1510-CourseProject/blob/master/Results/mi_gnn_dd.png" align="center" width="500"><figcaption>Fig.3 -  Information Plane for Graph Neural Network(tanh activation) trained on DD</figcaption></a>  

<img src="https://github.com/Vishwesh4/MAT1510-CourseProject/blob/master/Results/mi_gnn_infomax.png" align="center" width="500"><figcaption>Fig.4 -  Information Plane for Deep Graph Infomax (InfoGraph model) trained on DD</figcaption></a>     