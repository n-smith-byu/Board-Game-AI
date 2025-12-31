from src.board_game.players import AIPlayer
from .q_model import PhotosynthesisQModel
from photosynthesis.game import PhotosynthesisGame
from photosynthesis.game_board import GameBoard, PlayerGameState
from photosynthesis.actions import *
from photosynthesis.players.ai import RandomPlayer

from collections import defaultdict
from tqdm import tqdm
import random
import os
from enum import IntEnum

import numpy as np
import torch
import torch.nn.init as init
from torch.optim import Adam

class Actions(IntEnum):
    PASS = 0
    BUY_SEED = 1
    BUY_TREE_1 = 2
    BUY_TREE_2 = 3
    BUY_TREE_3 = 4
    PLANT = 5
    GROW = 6
    HARVEST = 7
    INIT = 8
 
class PhotosynthesisRLPlayer(AIPlayer):
    def __init__(self, num_players, player_num=None, temperature=0,
                 directory=os.path.join('src','photosynthesis','players','ai','deep_q_learning','models')):
        super(PhotosynthesisRLPlayer, self).__init__(player_num)

        self.game_num_players = num_players
        self.coords_to_node_ind_map = None
        self.node_ind_to_coords_map = None
        self.adj_mat = None
        self.adj_mat_2 = None
        self.adj_mat_3 = None
        self.turn_history = []

        self.map_board_spaces()
        self.create_adj_matrix()

        self.q_model = PhotosynthesisQModel(num_players)
        self.initialize_model_weights()
        self.q_model.eval()

        self.directory = directory
        self.training = False
        self.first_move_of_turn = True
        self.state_trails = defaultdict(list)

        self.chance_of_random_player = 0.9
        self.temperature = temperature
        self.epsilon = 0.5

    def map_board_spaces(self):
        board = GameBoard.get_soil_richness()
        self.coords_to_node_ind_map = {}
       
        k = -1
        for i in range(7):
            for j in range(7):
                if board[i][j] > 0:
                    k += 1
                    self.coords_to_node_ind_map[(i,j)] = k  
                    self.node_ind_to_coords_map[k] = (i,j)

        return
    
    def create_adj_matrix(self):
        self.adj_mat = self._get_adj_matrix(directions='all')
        self.adj_mat_2 = self.adj_mat @ self.adj_mat
        self.adj_mat_3 = self.adj_mat_2 @ self.adj_mat
        return
    
    def _get_adj_matrix(self, directions):
        if directions == 'all':
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
    
    # - - Generating State Vector - - 

    def generate_state_vec(self, state: PlayerGameState):
        # Starting inds for each section of the state vector
        FIRST_PLAYER_TOKEN = 0           # who has the first player token that round. 
        PLAYER_TREES = FIRST_PLAYER_TOKEN + self.game_num_players     # each player's trees on the board (encoded by size)

        TREE_GROWTH = PLAYER_TREES + self.game_num_players * 4      # whether each player has a tree/seed in 
                                                                     # their inventory that can grow a certain 
                                                                     # space.
        SPACE_MOVE_USED = TREE_GROWTH + self.game_num_players       # whether a space's action has already been used 
                                                                     # this turn.
        SUN_CHANNELS = SPACE_MOVE_USED + 1            # which spaces get sun the next two turns.

        num_channels = SUN_CHANNELS + 2
        state_vec = torch.zeros((37, num_channels))

        self._calc_first_player_token(state, state_vec, start_ind=FIRST_PLAYER_TOKEN)
        self._calc_player_trees(state, state_vec, trees_start_ind=PLAYER_TREES)
        self._calc_possible_tree_growth(state, state_vec, start_ind=TREE_GROWTH)

        return state_vec
    
    def _calc_first_player_token(self, state: PlayerGameState, state_vec, start_ind):
        # Calc position of the player with the first player token relative to this AIPlayer
        offset = self.player_num
        
        relative_first_player_token = (state.first_player_token - offset) % self.game_num_players
        state_vec[start_ind + relative_first_player_token] = 1.0

    def _calc_player_trees(self, state: PlayerGameState, state_vec, start_ind):
        tree_board = state.tree_board
        player_board = state.player_positions
        offset = self.player_num

        for (i,j), board_space in self.coords_to_node_ind_map.items():
            tree_size = tree_board[i, j]
            if tree_size >= 0:
                player_ind = player_board[i,j]
                player_relative_ind = (player_ind - offset) % self.game_num_players
                state_vec[board_space][start_ind + 4*player_relative_ind + tree_size] = 1.0
    
    def _calc_possible_tree_growth(self, state: PlayerGameState, state_vec, start_ind):
        tree_board = state.tree_board
        player_board = state.player_positions
        offset = self.player_num

        player_has_seeds = state.player_inventories[player_ind][0] > 0
        for (i,j), board_space in self.coords_to_node_ind_map.items():
            tree_size = tree_board[i,j]
            if tree_size >= 0:                  # if player has a tree there
                player_ind = player_board[i,j]
                if tree_size == 3 or state.player_inventories[player_ind][tree_size + 1] > 0:
                    player_relative_ind = (player_ind - offset) % self.game_num_players
                    state_vec[board_space][start_ind + player_relative_ind] = 1.0
            elif tree_size == -1 and player_has_seeds:       # if an empty board space and player has seeds
                # board spaces one, two, or three away
                ones = self.adj_mat[:, board_space]     
                twos = self.adj_mat_2[:, board_space]
                threes = self.adj_mat_3[:, board_space]

                for k in range(len(ones)):
                    can_plant_seed = False
                    i_, j_ = self.node_ind_to_coords_map[k]
                    if ones[k] > 0 and tree_board[i_, j_] >= 1:
                        can_plant_seed = True
                    elif twos[k] > 0 and tree_board[i_, j_] >= 2:
                        can_plant_seed = True
                    elif threes[k] > 0 and tree_board[i_, j_] == 3:
                        can_plant_seed = True
                    
                    if can_plant_seed:
                        player_ind = player_board[i_, j_]
                        player_relative_ind = (player_ind - offset) % self.game_num_players
                        state_vec[k][start_ind + player_relative_ind] = 1.0

    def _calc_space_move_used(self, state_vec, start_ind):
        used_spaces = []
        for action in self.turn_history:
            if isinstance(action, PlantSeed):
                used_spaces.append(action.position)
            elif isinstance(action, (GrowTree, HarvestTree)):
                used_spaces.append(action.tree.position)

        for coords in used_spaces:
            board_space_ind = self.coords_to_node_ind_map[coords]
            state_vec[board_space_ind][start_ind] = 1.0

    def _calc_expected_sunlight(self, state: PlayerGameState, state_vec, start_ind):
        for i in range(2):
            sun_pos = state.sun_pos + i + 1
            shadows = self._simulate_shadows(state.tree_board, sun_pos)
            state_vec[:, start_ind + 1] = 1.0       # assume everywhere gets sun to start
            for (i,j), board_space_ind in self.coords_to_node_ind_map.items():
                if shadows[i,j] == 1:
                    state_vec[board_space_ind][start_ind + i] = 0.0     # space expected to be in shadow in (i + 1) turns

    def _simulate_shadows(self, tree_board, sun_pos):
        sun_directions = GameBoard.get_sun_direction_vecs()
        shadows = GameBoard.get_empty_board(zeros=True)
        for (i,j), board_space_ind in self.coords_to_node_ind_map.items():
            if tree_board[i,j] > 0:
                this_tree_pos = np.array([i,j])
                sun_direction = sun_directions[sun_pos]
                for num_spaces_away in [1,2,3]:
                    neighbor_space = tuple(this_tree_pos - num_spaces_away*sun_direction)
                    if neighbor_space not in self._valid_board_spaces:
                        break

                    neighbor_tree_size = self._tree_board[*neighbor_space]
                    if neighbor_tree_size >= num_spaces_away and neighbor_tree_size >= tree.size:
                        return True
            
        return False
                        

    def map_action_to_vector(self, action):
        """
        Takes an action as input and creates a vector representing that action to pass into the Q model. 

        Uses a one-hot encoding to describe the kind of action,
        then two additonal one-hot encodings; one to describe which board space 
        the action is being applied from (e.g the parent tree) and which board space 
        the action is being applied to (e.g. where the seed is being planted), if
        applicable. 

        """
        buy_plant_grow_harvest_pass_actions = torch.zeros((9,))         # one-hot-encoding, see Actions enum
        action_board_space_and_recipient_space = torch.zeros((37,2))
        if isinstance(action, InitialPlacement):
            buy_plant_grow_harvest_pass_actions[Actions.INIT] = 1.0
            pos_ind = self.coords_to_node_ind_map[action.position]
            action_board_space_and_recipient_space[pos_ind, 1] = 1.0
        elif isinstance(action, BuyTree):
            buy_plant_grow_harvest_pass_actions[Actions.BUY_SEED + action.size] = 1.0
        elif isinstance(action, PlantSeed):
            buy_plant_grow_harvest_pass_actions[Actions.PLANT] = 1.0
            parent_pos_ind = self.coords_to_node_ind_map[action.parent_tree.position]
            seed_pos_ind = self.coords_to_node_ind_map[action.position]
            action_board_space_and_recipient_space[parent_pos_ind, 0] = 1.0
            action_board_space_and_recipient_space[seed_pos_ind, 1] = 1.0
        elif isinstance(action, GrowTree):
            buy_plant_grow_harvest_pass_actions[Actions.GROW] = 1.0
            space_ind = self.coords_to_node_ind_map[action.tree.position]
            action_board_space_and_recipient_space[space_ind, :] = 1.0
        elif isinstance(action, HarvestTree):
            buy_plant_grow_harvest_pass_actions[Actions.HARVEST] = 1.0
            space_ind = self.coords_to_node_ind_map[action.tree.position]
            action_board_space_and_recipient_space[space_ind, :] = 1.0
        elif isinstance(action, PassTurn):
            buy_plant_grow_harvest_pass_actions[Actions.PASS] = 1.0

        return torch.concat([buy_plant_grow_harvest_pass_actions,
                             action_board_space_and_recipient_space.flatten()], dim=0)
    

    # - - Model Initialization and Training - - 

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
    
    def initialize_model_weights(self):
        for param in self.q_model.parameters():
            if param.dim() > 1:  # Only initialize weights, not biases
                init.xavier_normal_(param)
            else:
                init.zeros_(param)  # Initialize biases to 0

    def save_model(self, file_name):
        """save to a file in the model's current directory"""
        path = os.path.join(self.directory, file_name)
        torch.save(self.q_model.state_dict(), path)

    def load_model(self, file_name):
        path = os.path.join(self.directory, file_name)
        self.q_model.load_state_dict(torch.load(path))
        self.q_model.eval()

    def train(self, N, gamma = 0.8, epsilon= 0.5, optimizer=Adam, save_every=20):
        self.epsilon = epsilon
        self.training = True
        self.q_model.train()
        optimizer = Adam(self.q_model.parameters())
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

        self.q_model.eval()
        self.training = False

    # - - 

    def choose_move(self, state: PlayerGameState):
        actions = torch.stack([self.map_action_to_vector(action) \
                             for action in state.available_actions])    # batch actions together
        
        # calculate Q vals
        state_vec = self.generate_state_vec(state)
        Q_vals = self.q_model(state_vec, actions)

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
        
        # track choices made this turn
        choice_ind = choice.item()
        action = state.available_actions[choice_ind]
        if isinstance(action, PassTurn):
            self.turn_history.clear()
        else:
            self.turn_history.append(action)

        return choice_ind


