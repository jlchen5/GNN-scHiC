import torch
from torch_geometric.data import Data, Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load your Hi-C data
df = pd.read_csv('/mnt/data/GSM6081068_Cell_ID_15.hic.txt', sep='\t')

# Assuming 'chr_A' and 'chr_B' are categories, we convert them to numeric codes to serve as node identifiers
chr_a = pd.Categorical(df['chr_A']).codes
chr_b = pd.Categorical(df['chr_B']).codes

# Create a list of tuples representing the edges (interactions) between regions
edges = list(zip(chr_a, chr_b))

# Convert edge list to tensor
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# Optional: create edge features based on 'num_obs', you might want to scale/normalize this
edge_features = torch.tensor(df['num_obs'].values, dtype=torch.float).view(-1, 1)
edge_features = (edge_features - edge_features.mean()) / edge_features.std()

# Optional: create node features, here just using one-hot encoding as placeholder
node_features = torch.eye(len(df['chr_A'].unique()))

# Create PyTorch Geometric data object
data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)

# Define your GNN model
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(node_features.shape[1], 16)
        self.conv2 = GCNConv(16, 32)
        self.fc = torch.nn.Linear(32, 1)  # Predicting 'num_obs', hence output is 1

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)

        return self.fc(x)

model = GCN()

# Define loss function and optimizer
criterion = torch.nn.MSELoss()  # Change depending on your needs
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model (placeholder code, implement actual training loop)
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.y)  # Define data.y as your target features
    loss.backward()
    optimizer.step()
    return loss

# Placeholder loop - replace with actual epochs and data splits
for epoch in range(200):
    loss = train()
    print(f'Epoch {epoch}, Loss {loss.item()}')
