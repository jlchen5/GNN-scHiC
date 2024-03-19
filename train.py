import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np

# Function to create a graph from Hi-C data
def create_graph(hi_c_data):
    """
    Convert Hi-C interaction data into graph format.
    Each genomic region is a node, and interactions are edges.
    Placeholder function - needs actual implementation based on your Hi-C data.
    """
    # For simplicity, let's assume `hi_c_data` is a list of tuples (node1, node2, interaction_strength)
    nodes = set()
    edge_index = []
    edge_attr = []
    
    for node1, node2, interaction_strength in hi_c_data:
        nodes.add(node1)
        nodes.add(node2)
        edge_index.append([node1, node2])
        edge_attr.append(interaction_strength)
    
    # Convert to PyTorch tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
    node_features = torch.randn((len(nodes), node_feature_size))  # Placeholder for actual features
    
    # Create graph data
    graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    return graph_data

# Define a simple GNN model
class HiCGNN(torch.nn.Module):
    def __init__(self):
        super(HiCGNN, self).__init__()
        self.conv1 = GCNConv(node_feature_size, 16)
        self.conv2 = GCNConv(16, 32)
        self.fc = torch.nn.Linear(32, 1)  # Adjust the output size based on your task

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = torch.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = torch.relu(self.conv2(x, edge_index, edge_attr=edge_attr))
        x = global_mean_pool(x, batch=data.batch)  # If you have batched data
        x = self.fc(x)
        return x

# Main program
if __name__ == "__main__":
    # Placeholder for actual Hi-C data
    hi_c_data = [(0, 1, 0.5), (1, 2, 0.8), (2, 3, 0.3)]  # Example data
    graph_data = create_graph(hi_c_data)

    model = HiCGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()  # Adjust based on your task

    # Dummy training loop - replace with actual training logic
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(graph_data)
        loss = criterion(out, torch.tensor([[1.0], [0.0], [0.5]]))  # Dummy target, replace with actual
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
