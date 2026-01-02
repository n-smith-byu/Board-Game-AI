from .state_tensor import PlayerStateTensor

import torch
import torch.nn as nn

class CustomGraphCNN(nn.Module):
    def __init__(self, in_dim, out_dim, A):
        super(CustomGraphCNN, self).__init__()
        
        A_tilde = A + torch.eye(A.shape[0])
        D_tilde = torch.sum(A_tilde, axis=1).reshape((-1, 1))
        self.register_buffer('A_norm', A_tilde / D_tilde, persistent=False)

        self.W = nn.Linear(in_dim, out_dim, dtype=torch.float32)
        self.W_self = nn.Linear(in_dim, out_dim, dtype=torch.float32)

    def forward(self, X):
        conv = self.A_norm @ X

        return self.W(conv) + self.W_self(X)


class PhotosynthesisQModel(nn.Module):

    def __init__(self, num_players, num_turns=18):
        super(PhotosynthesisQModel, self).__init__()

        self.num_players = num_players
        self.num_turns = num_turns
        STATE_DIM = 24*num_players + 10
        ACTION_DIM = 78

        # state encoder
        self.state_input = nn.Linear(in_features=STATE_DIM, out_dim=128)
        self.board_gcn_1 = CustomGraphCNN(in_dim=STATE_DIM, out_dim=128)
        self.board_gcn_2 = CustomGraphCNN(in_dim=128, out_dim=128)
        self.board_gcn_3 = CustomGraphCNN(in_dim=128, out_dim=128)
        self.board_gcn_4 = CustomGraphCNN(in_dim=128, out_dim=128)
        self.board_gcn_5 = CustomGraphCNN(in_dim=128, out_dim=128)
        self.state_final = nn.Linear(in_features=128, out_features=64)

        self.bn_state_in = nn.BatchNorm1d(num_features = 128)
        self.bn_gcn_1 = nn.BatchNorm1d(num_features = 128)
        self.bn_gcn_2 = nn.BatchNorm1d(num_features = 128)
        self.bn_gcn_3 = nn.BatchNorm1d(num_features = 128)
        self.bn_gcn_4 = nn.BatchNorm1d(num_features = 128)
        self.bn_gcn_5 = nn.BatchNorm1d(num_features = 128)
        self.bn_state_f = nn.BatchNorm1d(num_features = 64)

        # action encoder (each state has a batch of actions for all possible actions)
        self.action_input = nn.Linear(in_features=ACTION_DIM, out_features=32)
        self.bn_action_in = nn.BatchNorm1d(num_features = 32)

        # combined
        self.lin5 = nn.Linear(in_features=96, out_features=32)
        self.lin6 = nn.Linear(in_features=32, out_features=1)

        self.sigma = nn.LeakyReLU()


    def forward(self, state: PlayerStateTensor, actions: torch.Tensor):
        device = self.board_gcn_1.A_norm.device
        state_vec = state.get_tensor()
        state_vec.device(device)

        state_vec = self.sigma(self.bn_state_in(self.state_input(state)))
        
        # GCN 1 with Skip
        identity = state_vec
        state_vec = self.sigma(self.bn_gcn_1(self.board_gcn_1(state_vec)))
        state_vec = state_vec + identity # Residual connection
        
        # GCN 2 with Skip
        identity = state_vec
        state_vec = self.sigma(self.bn_gcn_2(self.board_gcn_2(state_vec)))
        state_vec = state_vec + identity

        # GCN 3 with Skip
        identity = state_vec
        state_vec = self.sigma(self.bn_gcn_3(self.board_gcn_3(state_vec)))
        state_vec = state_vec + identity

        # GCN 4 with Skip
        identity = state_vec
        state_vec = self.sigma(self.bn_gcn_4(self.board_gcn_4(state_vec)))
        state_vec = state_vec + identity

        # GCN 5 with Skip
        identity = state_vec
        state_vec = self.sigma(self.bn_gcn_5(self.board_gcn_5(state_vec)))
        state_vec = state_vec + identity

        # State Embedding Final
        state_vec = self.sigma(self.bn_state_f(self.state_final(state_vec)))

        # Action Embedding Final
        state_batch, action_batch, num_features = actions.shape
        action_vec = self.sigma(self.bn_action_in(self.action_input(actions)))

        # TODO: combine state and actions to predict Q-Values

        



        


        


        
