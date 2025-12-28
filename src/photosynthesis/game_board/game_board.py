from .Inventory import PlayerInventory
from .Store import PlayerStore
from .Trees import Tree
from .Points import PointsBank
from .Exceptions import *
from src.Photosynthesis.ActionTypes import BuyTree, PlantSeed, GrowTree, HarvestTree, PassTurn

from collections import defaultdict
import numpy as np

class GameBoard:
    @staticmethod
    def get_empty_board(zeros=False):
        F = np.inf
        board = np.array([[-1, -1, -1, -1, -F, -F, -F],
                          [-1, -1, -1, -1, -1, -F, -F],
                          [-1, -1, -1, -1, -1, -1, -F],
                          [-1, -1, -1, -1, -1, -1, -1],
                          [-F, -1, -1, -1, -1, -1, -1],
                          [-F, -F, -1, -1, -1, -1, -1],
                          [-F, -F, -F, -1, -1, -1, -1]])
        
        if zeros:
            board += 1
        
        return board
    
    @staticmethod
    def get_soil_richness():
        F = np.inf
        return np.array([[ 1,  1,  1,  1, -F, -F, -F],
                         [ 1,  2,  2,  2,  1, -F, -F],
                         [ 1,  2,  3,  3,  2,  1, -F],
                         [ 1,  2,  3,  4,  3,  2,  1],
                         [-F,  1,  2,  3,  3,  2,  1],
                         [-F, -F,  1,  2,  2,  2,  1],
                         [-F, -F, -F,  1,  1,  1,  1]])
    
    @staticmethod
    def get_sun_direction_vecs():
        return np.array([[ 1, 1],
                         [ 1, 0],
                         [ 0,-1],
                         [-1,-1],
                         [-1, 0],
                         [ 0, 1]])

    def __init__(self, num_players):
        self.__num_players = num_players
        self.reset()
        self.__valid_board_spaces = {(i,j) for i in range(self.__board_height) for j in range(self.__board_width) \
                                     if self.__tree_board[i,j] > -np.inf}


    def reset(self):
        self.__tree_board = GameBoard.get_empty_board()
        self.__player_board = GameBoard.get_empty_board()
        self.__points_bank = PointsBank()
        self.__trees:set[Tree] = set()        # set of all trees on the board
        self.__sun_position = -1

        self.__board_height = self.__tree_board.shape[0]
        self.__board_width = self.__tree_board.shape[1]

        self.__soil_richness = GameBoard.get_soil_richness()

        self.__players = defaultdict(dict)
        for player_ind in range(self.__num_players):
            self.__players[player_ind]['inventory'] = PlayerInventory(player_ind)
            self.__players[player_ind]['store'] = PlayerStore(player_ind)
            self.__players[player_ind]['suns'] = 0
            self.__players[player_ind]['points'] = 0
            self.__players[player_ind]['new_suns'] = 0
    
    def get_valid_seed_positions(self, tree:Tree):
        if tree not in self.__trees:
            raise ValueError('Tree does not exist on board')
        if tree.grown_this_turn():
            return set()
        
        valid_seed_positions = set()
        self.__find_valid_seed_positions(curr_position=tree.position,
                                         curr_distance=0,
                                         tree_size=tree.size,
                                         valid_seed_positions=valid_seed_positions)
        return valid_seed_positions

    def __find_valid_seed_positions(self, curr_position, curr_distance, tree_size, 
                                    valid_seed_positions:set):
        if curr_position not in self.__valid_board_spaces:
            return

        if self.__tree_board[*curr_position] == -1:
            valid_seed_positions.add(curr_position)

        if curr_distance == tree_size:
            return
        for direction in GameBoard.get_sun_direction_vecs():
            new_position = tuple(int(x) for x in (np.array(curr_position) + direction))
            self.__find_valid_seed_positions(curr_position=new_position, 
                                             curr_distance=curr_distance + 1,
                                             tree_size=tree_size,
                                             valid_seed_positions=valid_seed_positions)
        
    def get_trees_with_possible_actions(self, player_num, sizes=None):
        if sizes is None:
            sizes = {0,1,2,3}

        return {tree for tree in self.__trees if tree.player == player_num \
                     and tree.size in sizes \
                     and not tree.grown_this_turn()}
        
    def tree_in_shadow(self, tree:Tree):
        this_tree_pos = np.array(tree.position)
        sun_direction = self.get_sun_direction_vec()
        for num_spaces_away in [1,2,3]:
            neighbor_space = tuple(this_tree_pos - num_spaces_away*sun_direction)
            if neighbor_space not in self.__valid_board_spaces:
                break

            neighbor_tree_size = self.__tree_board[*neighbor_space]
            if neighbor_tree_size >= num_spaces_away and neighbor_tree_size >= tree.size:
                return True
            
        return False
    
    def __collect_suns(self):
        for player in self.__players:
            self.__players[player]['new_suns'] = 0

        tree: Tree = None
        for tree in self.__trees:
            if not self.tree_in_shadow(tree):
                self.__players[tree.player]['new_suns'] += tree.size
                new_sun_count = self.__players[tree.player]['suns'] + tree.size
                self.__players[tree.player]['suns'] = min(new_sun_count, 20)

    def get_open_board_spaces(self):
        return [board_space for board_space in self.__valid_board_spaces \
                if self.__tree_board[*board_space] == -1]

    def get_valid_board_spaces(self):
        return {board_space for board_space in self.__valid_board_spaces}

    def is_valid_board_space(self, pos):
        return tuple(pos) in self.__valid_board_spaces

    def rotate_sun(self):
        self.__sun_position = (self.__sun_position + 1) % 6

        for tree in self.__trees:
            tree.clear_grown_flag()

        self.__collect_suns()

    def __add_tree(self, tree:Tree, pos):
        if pos not in self.__valid_board_spaces:
            raise InvalidBoardSpaceException(pos, self.__valid_board_spaces)
        
        if self.__tree_board[*pos] > -1:
            message = f'Cannot place a tree at position {pos} because it is already occupied by another tree.'
            raise SpaceUnavailableException(message)
        
        self.__tree_board[*pos] = tree.size
        self.__player_board[*pos] = tree.player
        tree.set_position(pos)
        self.__trees.add(tree)

    def __remove_tree(self, tree:Tree):
        if tree in self.__trees:
            position = tree.position
            self.__trees.remove(tree)
            self.__players[tree.player]['store'].restock_tree(tree)
            self.__tree_board[*position] = -1
            self.__player_board[*position] = -1

            tree.clear_position()
            tree.clear_grown_flag()

            return tree
        else:
            raise ValueError('Tree does not exist on board')
        
    def get_possible_first_turn_spaces(self):
        return [(0,0), (0,1), (0,2), (0,3), (1,4), (2,5),
                (3,6), (4,6), (5,6), (6,6), (6,5), (6,4),
                (6,3), (5,2), (4,1), (3,0), (2,0), (1,0)]

    def get_possible_actions(self, player_num):
        possible_actions = set()
        player = self.__players[player_num]

        # get actions involving trees
        player_trees = {tree for tree in self.__trees if tree.player is player_num}
        for tree in player_trees:
            # check growing/harvesting
            if tree.size < 3:
                if player['inventory'].num_available_trees(tree.size + 1) > 0 \
                        and player['suns'] >= tree.cost_to_grow() \
                        and not tree.grown_this_turn():
                    possible_actions.add(GrowTree(player_num, tree))
            else:
                if player['suns'] >= tree.cost_to_grow() \
                        and not tree.grown_this_turn():
                    possible_actions.add(HarvestTree(player_num, tree))

            # check planting seeds
            if tree.size > 0 and player['inventory'].num_available_trees(size=0) > 0 \
                    and player['suns'] >= Tree.COST_TO_PLACE[0] \
                    and not tree.grown_this_turn():
                
                for space in self.get_valid_seed_positions(tree):
                    possible_actions.add(PlantSeed(player_num, tree, space))

        # get actions involving buying trees
        for size in [0,1,2,3]:
            if player['suns'] >= player['store'].get_cost(size):
                possible_actions.add(BuyTree(player_num, size))

        # passing is always an option
        possible_actions.add(PassTurn(player_num))

        return sorted(possible_actions, key=lambda action: action.sort_key())
    
    # --------------- #
    # Get Board Info #
    # --------------- #
    def get_sun_pos(self):
        return self.__sun_position
    
    def get_sun_direction_vec(self):
        return GameBoard.get_sun_direction_vecs()[self.__sun_position]
    
    def get_player_board(self):
        return self.__player_board.copy()

    def get_tree_board(self):
        return self.__tree_board.copy()
    
    def get_remaining_points(self):
        return self.__points_bank.remaining_points

    # --------------- #
    # Get Player Info #
    # --------------- #
    def get_player_scores(self):
        scores = [0]*self.__num_players
        for player in self.__players:
            scores[player] = self.__players[player]['points']
        
        return scores
    
    def get_player_score(self, player):
        return self.__players[player]['points']
    
    def get_player_suns(self, player):
        return self.__players[player]['suns']
    
    def get_player_new_suns_this_turn(self, player):
        return self.__players[player]['new_suns']
    
    # ------------- #
    #### Apply Actions ####
    # ------------- # 
    def player_buy_tree(self, player_num, size):
        player = self.__players[player_num]
        cost = player['store'].get_cost(size)

        if cost == np.inf:           # none left
            raise OutOfStockException(f"There are no trees of size {size} available in Player {player_num}'s store")
        if player['suns'] < cost:
            raise InsufficientSunsException(f"{cost} suns required to buy a tree of size {size}. " \
                                            + f"Player {player_num} only has {player['suns']}.")
        
        tree = player['store'].buy_tree(size)
        player['inventory'].add_tree(tree)
        player['suns'] -= cost

    def player_grow_tree(self, player_num, tree:Tree):
        if player_num != tree.player:
            raise ValueError('Player num and tree.player must match. Cannot grow tree of another player.')
        
        if tree.size not in [0,1,2]:
            raise ValueError('Invalid Tree Size. Can Only grow trees of size 0, 1, or 2')
        
        player = self.__players[player_num]
        if player['inventory'].num_available_trees(tree.size + 1) == 0:
            raise OutOfStockException(f'Cannot grow a tree of size {tree.size} without a tree ' + \
                                      f'of size {tree.size + 1} in inventory. Please buy a tree first')
        
        cost = tree.cost_to_grow()
        if player['suns'] < cost:
            raise InsufficientSunsException('Not enough suns to place tree.')
        
        position = tree.position
        self.__remove_tree(tree)
        replacement_tree:Tree = player['inventory'].remove_tree(tree.size + 1)
        self.__add_tree(replacement_tree, position)
        replacement_tree.set_grown_flag()

        player['suns'] -= cost

    def player_harvest_tree(self, player_num, tree:Tree):
        if player_num != tree.player:
            raise ValueError('Player num and tree.player must match. Cannot grow tree of another player.')
        
        if tree.size != 3:
            raise ValueError('Tree size must be 3 in order to harvest')
        
        player = self.__players[player_num]
        
        cost = tree.cost_to_grow()          # should be 4
        if player['suns'] < cost:
            raise InsufficientSunsException('Not enough suns to harvest tree.')
        
        position = tree.position
        self.__remove_tree(tree)
        soil_richness = self.__soil_richness[*position]
        player['points'] += self.__points_bank.claim_points(soil_richness)

    def player_plant_seed(self, player_num, parent:Tree, position:tuple):
        if parent.player != player_num:
            raise ValueError("player_num must match player of parent. " + \
                             "Players cannot plant a seed from another player's tree.")
        
        if parent.grown_this_turn():
            raise TreeAlreadyUsedActionException('This tree has already been grown or had a seed planted ' + \
                                                 'from it this turn')
        
        # check player inventory and suns
        player = self.__players[player_num]

        if player['inventory'].num_available_trees(size=0) == 0:
            raise OutOfStockException('No seeds available to plant in inventory. Please buy a seed first.')
        
        cost = Tree.COST_TO_PLACE[0]
        if player['suns'] < cost:
            raise InsufficientSunsException('Not enough suns to plant a seed')
    
        # make sure the seed position is valid (takes the longets, so do last)
        if position not in self.get_valid_seed_positions(parent):
            raise ValueError("Cannot plant a seed on this board space. " +\
                             "Either the space doesn't exist, isn't reachable by this tree, or is occupied.")
        
        # plant the seed
        seed:Tree = player['inventory'].remove_tree(size=0)
        self.__add_tree(seed, position)
        seed.set_grown_flag()
        parent.set_grown_flag()
        player['suns'] -= cost

    def player_initial_tree_placement(self, player_num, position):
        possible_first_turn_spaces = self.get_possible_first_turn_spaces()
        if position not in possible_first_turn_spaces:
            raise ValueError('Not a valid board space for initial placement')
        
        if self.__tree_board[*position] > -1:
            raise ValueError('Board Space already occupied')
        
        player = self.__players[player_num]
        tree = player['inventory'].remove_tree(size=1)
        self.__add_tree(tree, position)

    def print_boards(self):
        print('Tree Sizes:')
        print(self.__tree_board, '\n')
        print('Player Occupancies')
        print(self.__player_board, '\n')




