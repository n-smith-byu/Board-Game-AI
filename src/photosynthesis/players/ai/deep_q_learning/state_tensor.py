from src.photosynthesis import PlayerGameState
from src.photosynthesis.game_board import GameBoard
from src.photosynthesis.actions import *

import numpy as np
import torch


class PlayerStateTensor:
    @classmethod
    def __init__(self, game_state: PlayerGameState, moves_this_turn,
                 coords_to_node_ind_map, node_ind_to_coords_map, 
                 adj_mat, adj_mat_2, adj_mat_3):
        # Define start inds for each section of the state tensor
        # (based on number of players n)
        self.num_players = game_state.num_players

        # marks who has the first player token this round. 
        # [n channels, one hot encoded]
        self.FIRST_PLAYER_TOKEN = 0        

        # each player's trees on the board (encoded by size)
        # [4n channels, encoded by player and tree size]
        self.PLAYER_TREES = self.FIRST_PLAYER_TOKEN + self.num_players

        # mark number of trees of each size available to each player in their inventory
        # as a proportion of total number of trees of that size\
        # [4n channels, encoded by player and tree size]
        self.PLAYER_INVENTORIES = self.PLAYER_TREES + self.num_players * 4     

        # mark number of trees of each size available to each player in their store
        # as a proportion of total number of trees of that size
        # [4n channels, encoded by player and tree size]
        self.PLAYER_STORES_RAW_NUM = self.PLAYER_INVENTORIES + self.num_players * 4  

        # encode for each player, the state of their store (non-linear pricing)
        # [13n channels, 4 seed, 4 small, 3 medium, 2 large]
        self.PLAYER_STORES_ENCODING = self.PLAYER_STORES_RAW_NUM + self.num_players*4

        # mark how many suns each player has out of total possible suns
        self.PLAYER_SUNS = self.PLAYER_STORES_ENCODING + self.num_players * 13

        # mark how many points a player has
        self.PLAYER_SCORES = self.PLAYER_SUNS + self.num_players

        # mark, for each space, if an action has already been done this turn
        # [1 channel]
        self.SPACE_MOVES_USED = self.PLAYER_SCORES + self.num_players

        # mark current position of the sun
        # [6 channels, one hot encoding]
        self.SUN_POS = self.SPACE_MOVES_USED + 1

        # mark which spaces expected to get sun the next two rounds (based only on current trees)
        # [2 channels, one per round] 
        self.SUNLIGHT_CHANNELS = self.SUN_POS + 6   

        # mark how many turns are left in the game, out of total rounds.
        self.GAME_ROUND = self.SUNLIGHT_CHANNELS + 2

        self.coords_to_node_ind_map: dict[tuple, int] = coords_to_node_ind_map
        self.node_ind_to_coords_map: dict[int, tuple] = node_ind_to_coords_map
        self.adj_mat = adj_mat
        self.adj_mat_2 = adj_mat_2
        self.adj_mat_3 = adj_mat_3

        num_channels = self.GAME_ROUND + 1
        self.state_tensor = torch.zeros((len(node_ind_to_coords_map), num_channels))

        # map players to inds relative to the current player
        self.player_map = {}
        for ind in range(game_state.num_players):
            relative_ind = (ind - game_state.player_num) % game_state.num_players
            self.player_map[ind] = relative_ind

        self._set_first_player_token(game_state.first_player_token)
        self._set_player_trees(game_state.tree_board, game_state.player_positions)
        self._set_player_inventory_and_store(
            game_state.player_inventories,
            game_state.player_stores,
            )
        self._set_player_suns(game_state.all_player_suns)
        self._set_player_scores(game_state.all_player_points)
        self._set_spaces_used(moves_this_turn)
        self._set_sun_pos(game_state.sun_pos)
        self._set_game_round(game_state.total_game_turns, game_state.remaining_turns)
        self._calc_expected_sunlight(
            game_state.tree_board, 
            game_state.sun_pos, 
            game_state.remaining_turns
            )

    # - - - - - - - - - - - - 

    def _set_first_player_token(self, first_player_token):
        channel = self.FIRST_PLAYER_TOKEN + self.player_map[first_player_token]
        self.state_tensor[:, channel] = 1.0

    def _set_player_trees(self, tree_board, player_board):
        for (i,j), board_space in self.coords_to_node_ind_map.items():
            tree_size = tree_board[i, j]
            if tree_size >= 0:
                player_ind = player_board[i,j]
                channel = self.PLAYER_TREES + 4*self.player_map[player_ind] + tree_size
                self.state_tensor[board_space][channel] = 1.0

    def _set_player_inventory_and_store(self, inventories, stores):
        tree_total_nums = {
            0: 6.0, 
            1: 8.0, 
            2: 4.0, 
            3: 2.0
            }
        store_total_num = {
            0: 4,
            1: 4,
            2: 3,
            3: 2
            }
        size_offsets = {
            0: 0,
            1: 4,
            2: 8,
            3: 11
        }
        for player_ind in range(self.num_players):
            for tree_size, total_num in tree_total_nums.items():
                player_offset = self.player_map[player_ind]
                # inventory
                channel = self.PLAYER_INVENTORIES + 4*player_offset + tree_size
                self.state_tensor[:, channel] = inventories[player_ind][tree_size] / total_num

                # store raw num
                channel = self.PLAYER_STORES_RAW_NUM + 4*player_offset + tree_size
                self.state_tensor[:, channel] = stores[player_ind][tree_size] / total_num

                # store_encoding
                for count in range(store_total_num[tree_size]):
                    if count < stores[player_ind][tree_size]:
                        size_count_offset = size_offsets[tree_size] + count
                        channel = self.PLAYER_STORES_ENCODING + 13*player_offset + size_count_offset
                        self.state_tensor[:, channel] = 1.0

    def _set_player_suns(self, player_suns):
        for player_ind in range(self.num_players):
            channel = self.PLAYER_SUNS + self.player_map[player_ind]
            self.state_tensor[:, channel] = player_suns[player_ind] / 20.0

    def _set_player_scores(self, player_scores):
        for player_ind in range(self.num_players):
            channel = self.PLAYER_SCORES + self.player_map[player_ind]
            self.state_tensor[:, channel] = player_scores[player_ind] / 10.0
    
    def _set_spaces_used(self, moves_this_turn):
        used_spaces = []
        for action in moves_this_turn:
            if isinstance(action, PlantSeed):
                used_spaces.append(action.position)
            elif isinstance(action, (GrowTree, HarvestTree)):
                used_spaces.append(action.tree.position)

        for coords in used_spaces:
            board_space_ind = self.coords_to_node_ind_map[coords]
            self.state_tensor[board_space_ind][self.SPACE_MOVES_USED] = 1.0

    def _set_sun_pos(self, sun_pos):
        channel = self.SUN_POS + sun_pos
        self.state_tensor[:, channel] = 1.0

    def _set_game_round(self, total_rounds, remaining_rounds):
        self.state_tensor[:, self.GAME_ROUND] = remaining_rounds / total_rounds

    def _calc_expected_sunlight(self, tree_board, sun_pos, remaining_turns):
        future_turns = remaining_turns - 1

        # channels are already 0, meaning no sun.
        # if near the end of game, then mark as no sun for 
        # rounds that are after the game ends

        for i in range(min(2, future_turns)):
            next_sun_pos = (sun_pos + i + 1) % 6
            channel = self.SUNLIGHT_CHANNELS + i

            shadows = self._simulate_shadows(tree_board, next_sun_pos)
            self.state_tensor[:, channel] = 1.0 - shadows  

    def _simulate_shadows(self, tree_board, sun_pos):
        sun_directions = GameBoard.get_sun_direction_vecs()
        sun_vec = sun_directions[sun_pos]
        
        shadows = np.zeros(len(self.node_ind_to_coords_map))

        for (i,j), _ in self.coords_to_node_ind_map.items():
            shadow_size = tree_board[i,j]
            if shadow_size > 0:
                start_pos = np.array([i,j])
                for dist in range(1, int(shadow_size) + 1):
                    neighbor_space = tuple(start_pos + dist*sun_vec)
                    if neighbor_space not in self.coords_to_node_ind_map:
                        break
                    neighbor_size = tree_board[*neighbor_space]
                    if neighbor_size <= shadow_size:
                        shadowed_space = self.coords_to_node_ind_map[neighbor_space]
                        shadows[shadowed_space] = 1.0
            
        return torch.tensor(shadows)
    
    def get_tensor(self):
        return self.state_tensor