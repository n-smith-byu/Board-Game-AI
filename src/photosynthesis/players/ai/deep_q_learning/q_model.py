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
        new_features = self.W(conv) + self.W_self(X)

        return new_features

class PhotosynthesisQModel(nn.Module):

    def __init__(self, num_players, num_turns=18):
        super(PhotosynthesisQModel, self).__init__()

        self.num_players = num_players
        self.num_turns = num_turns
        vec_dim = 4*num_players

        self.sigma = nn.LeakyReLU()

        self.player_tree_1 = CustomGraphCNN(in_dim=vec_dim, out_dim=vec_dim)
        self.player_tree_2 = CustomGraphCNN(in_dim=vec_dim, out_dim=vec_dim)
        self.player_tree_3 = CustomGraphCNN(in_dim=vec_dim, out_dim=vec_dim)
        self.player_tree_4 = nn.Linear(in_features=vec_dim, out_features=8)

        self.sun_influence_1 = CustomGraphCNN(in_dim=vec_dim, out_dim=vec_dim)
        self.sun_influence_2 = CustomGraphCNN(in_dim=vec_dim, out_dim=vec_dim)
        self.sun_influence_3 = CustomGraphCNN(in_dim=vec_dim, out_dim=vec_dim)
        self.sun_influence_4 = nn.Linear(in_features=vec_dim, out_features=8)

        self.lin3 = nn.Linear(in_features=8, out_features=8)
        self.lin4 = nn.Linear(in_features=8, out_features=1)

        sun_encoding_size = 6
        player_suns_size = 1
        remaining_turns_size = 1
        action_size = 78
        prev_output_size=74
        total_size = prev_output_size + sun_encoding_size + player_suns_size + \
            remaining_turns_size + num_players + action_size
        
        self.lin5 = nn.Linear(in_features=total_size, out_features=16)
        self.lin6 = nn.Linear(in_features=16, out_features=1)


    def forward(self, state, actions):
        (player_enc, player_trees_enc, tree_influence, sun_influence_next_turn, 
                sun_pos_enc, player_suns, remaining_turns_enc) = state

        x1 = self.player_tree_1(X = player_trees_enc, A = tree_influence)
        x1 = self.sigma(x1)
        x1 = self.player_tree_2(X = x1, A = tree_influence)
        x1 = self.sigma(x1)
        x1 = self.player_tree_3(X = x1, A = tree_influence)
        x1 = self.sigma(x1)
        x1 = self.player_tree_4(x1)

        x2 = self.sun_influence_1(X = player_trees_enc, A = sun_influence_next_turn)
        x2 = self.sigma(x2)
        x2 = self.sun_influence_2(X = x2, A = sun_influence_next_turn)
        x2 = self.sigma(x2)
        x2 = self.sun_influence_3(X = x2, A = sun_influence_next_turn)
        x2 = self.sigma(x2)
        x2 = self.sun_influence_4(x2)

        x3 = torch.concat([x1, x2], dim=0)
        x3 = self.lin3(x3)
        x3 = self.sigma(x3)

        x4 = self.lin4(x3)

        x4 = torch.concat([x4.flatten(), player_suns, sun_pos_enc, player_enc,
                          remaining_turns_enc], dim=0)
        x4 = torch.concat([x4.unsqueeze(0).expand(actions.shape[0], -1), actions], dim=1)
        x5 = self.lin5(x4)
        x6 = self.lin6(x5)

        return x6.flatten()



        


        


        
