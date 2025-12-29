from src.board_game.players import AIPlayer
import os
from collections import defaultdict
import random
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.init as init
from torch.optim import Adam

from .model import PhotosynthesisQModel
from photosynthesis.game import PhotosynthesisGame
from photosynthesis.game_board import GameBoard, BoardSummary
from photosynthesis.actions import *
from photosynthesis.players.ai import RandomPlayer
 
class PhotosynthesisRLPlayer(AIPlayer):
    def __init__(self, num_players, player_num=None, 
                 directory=os.path.join('src','Photosynthesis','RL','SavedModels')):
        super(PhotosynthesisRLPlayer, self).__init__(player_num)

        self.game_num_players = num_players
        self.board_space_map = None

        self.ones = None
        self.twos = None
        self.threes = None

        self.get_one_two_three_adj_mat()
        num_buy_actions = 4
        num_grow_harvest_plant_actions = torch.sum(self.threes)

        self.num_possible_actions = num_grow_harvest_plant_actions + num_buy_actions

        self.model = PhotosynthesisQModel(num_players, self.num_possible_actions)
        self.initialize_model_weights()
        self.model.eval()

        self.directory = directory
        self.training = False
        self.first_move_of_turn = True
        self.state_trails = defaultdict(list)

        self.chance_of_random_player = 0.9
        self.epsilon = 0.5


    def initialize_model_weights(self):
        for param in self.model.parameters():
            if param.dim() > 1:  # Only initialize weights, not biases
                init.xavier_normal_(param)
            else:
                init.zeros_(param)  # Initialize biases to 0


    def save_model(self, file_name):
        """save to a file in the model's current directory"""
        path = os.path.join(self.directory, file_name)
        torch.save(self.model.state_dict(), path)

    def load_model(self, file_name):
        path = os.path.join(self.directory, file_name)
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def map_board_spaces(self):
        if self.board_space_map is not None:
            return self.board_space_map

        board = GameBoard.get_soil_richness()
        self.board_space_map = {}
       
        k = -1
        for i in range(7):
            for j in range(7):
                if board[i][j] > 0:
                    k += 1
                    self.board_space_map[(i,j)] = k  

        return self.board_space_map


    def get_board_adjacency_matrix(self, directions:list=None):
        if directions is None:
            directions = (i for i in range(6))

        direction_vecs = [dir_vec for i, dir_vec in enumerate(GameBoard.get_sun_direction_vecs()) \
                          if i in directions]

        board_space_map = self.map_board_spaces()
        adj_mat = torch.zeros((37, 37))

        for space in board_space_map:
            space_ind = board_space_map[space]
            for dir in direction_vecs:
                new_space = tuple(int(x) for x in np.array(space) + dir)
                if new_space in board_space_map:
                    new_space_ind = board_space_map[new_space]
                    adj_mat[space_ind, new_space_ind] = 1

        return adj_mat
    
    def get_tree_specific_adj_mat(self, tree_board):

        ones, twos, threes = self.get_one_two_three_adj_mat()

        new_adj_mat = torch.zeros_like(ones)

        board_space_map = self.map_board_spaces()
        for space, k in board_space_map.items():
            if tree_board[*space] == 1:
                new_adj_mat[k,:] = ones[k, :]
            elif tree_board[*space] == 2:
                new_adj_mat[k,:] = twos[k, :]
            elif tree_board[*space] == 3:
                new_adj_mat[k,:] = threes[k, :]

        return new_adj_mat

    def get_one_two_three_adj_mat(self):
        if self.ones is None or self.twos is None or self.threes is None:
            self.ones = self.get_board_adjacency_matrix()
            self.twos = ((self.ones @ self.ones) > 0) * 1.0
            self.threes = ((self.twos @ self.ones) > 0) * 1.0

        return self.ones.clone(), self.twos.clone(), self.threes.clone()

    def map_action_to_vec(self, action):
        """
        Takes an action as input and creates a vector representing that action to pass into the Q model. 

        Uses a one-hot encoding to describe the kind of action,
        then two additonal one-hot encodings; one to describe which board space 
        the action is being applied from (e.g the parent tree) and which board space 
        the action is being applied to (e.g. where the seed is being planted), if
        applicable. 

        """
        board_space_map = self.map_board_spaces()
        buy_actions = torch.zeros((4,))
        grow_harvest_plant_actions = torch.zeros((37,2))
        if isinstance(action, BuyTree):
            buy_actions[action.size] = 1.0
        elif isinstance(action, GrowTree) or isinstance(action, HarvestTree):
            space_ind = board_space_map[action.tree.position]
            grow_harvest_plant_actions[space_ind, :] = 1.0
        elif isinstance(action, PlantSeed):
            parent_pos_ind = board_space_map[action.parent_tree.position]
            seed_pos_ind = board_space_map[action.position]
            grow_harvest_plant_actions[parent_pos_ind, 0] = 1.0
            grow_harvest_plant_actions[seed_pos_ind, 1] = 1.0
        elif isinstance(action, InitialPlacement):
            pos_ind = board_space_map[action.position]
            grow_harvest_plant_actions[pos_ind, 1] = 1.0

        return torch.concat([grow_harvest_plant_actions.flatten(), buy_actions], dim=0)

    def generate_state_vec(self, state:BoardSummary):
        # one-hot-encoding curr_player_turn
        player_enc = torch.zeros(self.game_num_players)
        player_enc[state.player_num] = 1.0

        # one-hot-encode tree/player positions
        board_space_map = self.map_board_spaces()
        player_trees_enc = torch.zeros((37, self.game_num_players*4))
        for space, k in board_space_map.items():
            tree_size = state.tree_board[*space]

            if tree_size >= 0:
                player = state.player_positions[*space]
                player_trees_enc[k, int(player*4 + tree_size)] = 1.0

        # get some adjacency matrices
        tree_influence = self.get_tree_specific_adj_mat(state.tree_board)
        sun_influence_next_turn = self.get_board_adjacency_matrix(directions=[(state.sun_pos + 1) % 6])

        # encode sun position this turn
        sun_pos_enc = torch.zeros((6,))
        sun_pos_enc[state.sun_pos] = 1.0


        player_suns = torch.tensor([state.player_suns])

        # encode remaining_turns
        remaining_turns_enc = torch.tensor([state.remaining_turns / state.total_game_turns])

        return (player_enc, player_trees_enc, tree_influence, sun_influence_next_turn, 
                sun_pos_enc, player_suns, remaining_turns_enc)
    
    def _choose_players(self):
        available_models = os.listdir(self.directory)
        if len(available_models) == 0:
            self.save_model('model_0.pth')

        players = []
        my_player_num = np.random.choice(self.game_num_players)
        for i in range(self.game_num_players):
            if i == my_player_num:
                players.append(self)
            elif np.random.uniform(0,1) < 0.9 / (self.game_num_players - 1):
                players.append(self)
            else:
                if np.random.uniform(0,1) < self.chance_of_random_player:
                    players.append(RandomPlayer(player_num=i))
                    self.chance_of_random_player = self.chance_of_random_player*0.99
                else:
                    other = PhotosynthesisRLPlayer(self.game_num_players, player_num=i,
                                                   directory=self.directory)
                    other.load_model(random.choice(available_models))
                    players.append(other)

        return players

    def train(self, N, gamma = 0.8, epsilon= 0.5, optimizer=Adam, save_every=20):
        self.epsilon = epsilon
        self.training = True
        self.model.train()
        optimizer = Adam(self.model.parameters())
        game = None
        losses = []
        for n in tqdm(range(N), desc='Training...'):
            players = self._choose_players()
            game = PhotosynthesisGame(players)

            optimizer.zero_grad()
            final_scores = game.run()
            loss = torch.tensor([0.0], dtype=torch.float32)
            k = 0
            for player_num, trace in self.state_trails.items():
                curr_score = 0
                for i in range(len(trace)):
                    _, _, Q_curr, choice = trace[i]
                    if i == len(trace) - 1:
                        reward = final_scores[player_num] - max(final_scores)
                        target = reward
                    else:
                        new_score, new_suns, Q_next, _ = trace[i + 1]
                        score_diff = new_score - curr_score
                        curr_score = new_score

                        reward = score_diff + new_suns / 3              # reward points for harvesting trees
                        target = reward + gamma * torch.max(Q_next)
                    
                    loss += (Q_curr[choice] - target)**2
                    k += 1

            loss /= k
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            self.state_trails = defaultdict(list)
            self.epsilon = self.epsilon*0.999

            if n != 0 and n % save_every == 0:
                model_num = len(os.listdir(self.directory))
                self.save_model(f'model_{model_num}.pth')

        self.model.eval()
        self.training = False


    def choose_move(self, state: BoardSummary):
        X = torch.stack([self.map_action_to_vec(action) \
                             for action in state.available_actions])    # batch actions together
        
        # calculate Q vals
        state_vec = self.generate_state_vec(state)
        Q_vals = self.model(state_vec, X)

        # make a choice (epsilon greedy if training)
        if self.training and np.random.uniform(0,1) < self.epsilon:
            choice = torch.tensor([np.random.choice(len(state.available_actions))])

        else:
            choice = torch.argmax(Q_vals)

        if self.training:
            if self.first_move_of_turn:
                new_suns_from_last_action = state.player_new_suns
                self.first_move_of_turn = False
            else:
                new_suns_from_last_action = 0
            
            if len(state.available_actions) == 1:
                self.first_move_of_turn = True      # if only option is to pass, reset the flag for next turn

            self.state_trails[state.player_num].append((new_suns_from_last_action, state.player_score, 
                                                        Q_vals, choice))
        
        return choice.item()


