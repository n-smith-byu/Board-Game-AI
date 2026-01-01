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
    
class PhotoynthesisValuePolicyModel(torch.Module):

    def __init__(self, in_channels):
        super().__init__()

        _feature_size = 128

        self.board_gcn_1 = GraphCNN(in_dim=in_channels, out_dim=_feature_size)
        self.board_gcn_2 = GraphCNN(in_dim=_feature_size, out_dim=_feature_size)
        self.board_gcn_3 = GraphCNN(in_dim=_feature_size, out_dim=_feature_size)
        self.board_gcn_4 = GraphCNN(in_dim=_feature_size, out_dim=_feature_size)
        self.board_gcn_5 = GraphCNN(in_dim=_feature_size, out_dim=_feature_size)

        self.value_1 = nn.Linear(in_features=_feature_size, out_features=_feature_size)
        self.value_2 = nn.Linear(in_features=_feature_size, out_features=32)
        self.value_out = nn.Linear(in_features=_feature_size, out_features=1)

        self.policy_1 = nn.Linear(in_features=_feature_size, out_features=_feature_size)
        self.policy_2 = nn.Linear(in_features=_feature_size, out_features=_feature_size)
        self.policy_out = nn.Linear(in_features=_feature_size, out_features=_feature_size)

        self.sigma = nn.ReLU()

    def forward(self, board, adj_mat):
        x = self.board_gcn_1(board, adj_mat)
        x = self.sigma(x)
        x2 = self.board_gcn_2(board, adj_mat)
        x2 = self.sigma(x2)
        x3 = self.board_gcn_3(board, adj_mat)
        x3 = self.sigma(x3)

        return policy, val