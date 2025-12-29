import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphCNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GraphCNN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.W = nn.Linear(in_dim, out_dim, dtype=torch.float32)
        self.W_self = nn.Linear(in_dim, out_dim, dtype=torch.float32)

    def forward(self, X, A):
        A_tilde = A + torch.eye(A.shape[0])
        conv = A_tilde @ X / torch.sum(A_tilde, axis=1).reshape((A_tilde.shape[0],1))
        new_features = self.W(conv) + self.W_self(X)

        return new_features