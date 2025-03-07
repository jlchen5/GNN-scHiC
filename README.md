# GNN-scHiC

- `create_graph` is a function meant to transform Hi-C interaction data into a graph format compatible with PyTorch Geometric. You'll need to adjust this function to work with your actual data.
- HiCGNN is a simple two-layer GCN model. The real complexity and architecture of your model might differ based on your data and the specifics of your prediction task.
- The training loop is very basic and for illustration only. In a real scenario, you would split your data into training, validation, and test sets, and monitor performance across epochs more comprehensively.

First, ensure you have the necessary packages installed. You may need to install PyTorch and PyTorch Geometric:
~~~
pip install torch torchvision torchaudio
pip install torch-geometric
~~~


## License
This project is available under the [MIT License](https://github.com/jlchen5/GNN-scHiC/blob/main/LICENSE).
