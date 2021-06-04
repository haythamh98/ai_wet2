import math

import numpy as np

import logic
import random
from AbstractPlayers import *
import time

# commands to use for move players. dictionary : Move(enum) -> function(board),
# all the functions {up,down,left,right) receive board as parameter and return tuple of (new_board, done, score).
# new_board is according to the step taken, done is true if the step is legal, score is the sum of all numbers that
# combined in this step.
# (you can see GreedyMovePlayer implementation for example)
commands = {Move.UP: logic.up, Move.DOWN: logic.down,
            Move.LEFT: logic.left, Move.RIGHT: logic.right}


# generate value between {2,4} with probability p for 4
def gen_value(p=PROBABILITY):
    return logic.gen_two_or_four(p)


class GreedyMovePlayer(AbstractMovePlayer):
    """Greedy move player provided to you (no need to change),
    the player receives time limit for a single step and the board as parameter and return the next move that gives
    the best score by looking one step ahead.
    """

    def get_move(self, board, time_limit) -> Move:
        optional_moves_score = {}
        for move in Move:
            new_board, done, score = commands[move](board)  # do a run for the board, on trying the "move" direction
            if done:
                optional_moves_score[move] = score

        return max(optional_moves_score,
                   key=optional_moves_score.get)  # return comparing to their best value (aka get function)


class RandomIndexPlayer(AbstractIndexPlayer):
    """Random index player provided to you (no need to change),
    the player receives time limit for a single step and the board as parameter and return the next indices to
    put 2 randomly.
    """

    def get_indices(self, board, value, time_limit) -> (int, int):
        a = random.randint(0, len(board) - 1)
        b = random.randint(0, len(board) - 1)
        while board[a][b] != 0:
            a = random.randint(0, len(board) - 1)
            b = random.randint(0, len(board) - 1)
        return a, b


# optimization: big numbers close to wall
# part A
class ImprovedGreedyMovePlayer(AbstractMovePlayer):
    """Improved greedy Move Player,
    implement get_move function with greedy move that looks only one step ahead with heuristic.
    (you can add helper functions as you want).
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed

    def get_move(self, board, time_limit) -> Move:
        optional_moves_score = {}
        for move in Move:
            new_board, done, score = commands[move](board)  # do a run for the board, on trying the "move" direction
            if done:
                random.seed()
                # maximize score, while keep big numbers close to a wall, and prefer steps that increase the number of empty tiles
                empty_tiles = max(self.get_number_of_empty_tiles(new_board), 1)
                closely_score = self.get_closly_score(new_board)
                sum_all = max(self.sum_all_tiles(new_board), 1)
                corn_score = max(self.corner_score(new_board), 1)
                close_to_wall_scor = max(self.close_to_wall_score(new_board), 1)
                div_wi = 64
                aa, bb, bigges = self.get_biggest_tile_pos(new_board)
                AMP = math.log2(bigges)
                hur = empty_tiles / 16 + 1 * corn_score / (div_wi * sum_all) + (close_to_wall_scor / (div_wi * sum_all))
                # print(f'empty_tiles {empty_tiles} corn_score = {corn_score} close_to_wall_scor = {close_to_wall_scor}  ')
                optional_moves_score[move] = score + AMP * hur
                # print(f'score {score} hur = {hur} final = {optional_moves_score[move]}')
        print(optional_moves_score[max(optional_moves_score,
                                       key=optional_moves_score.get)])
        return max(optional_moves_score,
                   key=optional_moves_score.get)  # return comparing to their best value (aka get function)

    def popMin(self, my_dict):
        if len(my_dict) == 0:
            return
        mini = min(my_dict, key=my_dict.get)
        for key, value in my_dict.items():
            if value == mini:
                my_dict.pop(key)

    def close_to_wall_score(self, board) -> int:
        to_ret = 0
        for i in range(0, len(board)):
            to_ret += board[i][0]
            to_ret += board[i][-1]
        for i in range(1, len(board) - 1):
            to_ret += board[0][i]
            to_ret += board[-1][i]
        return to_ret

    def get_closly_score(self,
                         board) -> int:  # return 1 if the biggest number has value = x, and next to it tile with x/2
        i, j, val = self.get_biggest_tile_pos(board)
        if i == -1:
            return 0
        if i - 1 > 0:
            if board[i - 1][j] == val / 2:
                return 1
        if j - 1 > 0:
            if board[i][j - 1] == val / 2:
                return 1
        if i + 1 < len(board):
            if board[i + 1][j] == val / 2:
                return 1
        if j + 1 < len(board):
            if board[i][j + 1] == val / 2:
                return 1
        return 0

    def sum_all_tiles(self, board) -> int:
        to_ret = 0
        for i in range(0, len(board)):
            for j in range(0, len(board)):
                to_ret += board[i][j]
        return to_ret

    def corner_score(self, board) -> int:
        to_ret = 0
        to_ret += board[-1][0]
        to_ret += board[-1][-1]
        to_ret += board[0][-1]
        to_ret += board[0][0]
        return to_ret

    def get_number_of_empty_tiles(self, board) -> int:
        to_ret = 0
        for i in range(0, len(board)):
            for j in range(0, len(board)):
                if board[i][j] == 0:
                    to_ret += 1
        return to_ret

    def get_biggest_new_merge(self, board, old) -> int:
        to_ret = 0
        for i in range(0, len(board)):
            for j in range(0, len(board)):
                if board[i][j] == 0:
                    to_ret += 1
        return to_ret

    def get_number_of_tiles_attached_to_wall(self, board) -> int:
        to_ret = 0
        for i in range(0, len(board)):
            if board[i][0] != 0:
                to_ret += 1
            if board[i][-1] != 0:
                to_ret += 1
        for i in range(1, len(board) - 1):
            if board[0][i] != 0:
                to_ret += 1
            if board[-1][i] != 0:
                to_ret += 1
        return to_ret

    def get_biggest_tile_pos(self, board) -> (int, int, int):
        max_val = 0
        for i in range(0, len(board)):
            for j in range(0, len(board)):
                if board[i][j] > max_val:
                    max_val = board[i][j]
        for i in range(0, len(board)):
            for j in range(0, len(board)):
                if board[i][j] == max_val:
                    return i, j, max_val
        return -1, -1, 0


# part B
class MiniMaxMovePlayer(AbstractMovePlayer):
    """MiniMax Move Player,
    implement get_move function according to MiniMax algorithm
    (you can add helper functions as you want).
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        raise NotImplementedError

    # TODO: add here helper functions in class, if needed


class MiniMaxIndexPlayer(AbstractIndexPlayer):
    """MiniMax Index Player,
    this player is the opponent of the move player and need to return the indices on the board where to put 2.
    the goal of the player is to reduce move player score.
    implement get_indices function according to MiniMax algorithm, the value in minimax player value is only 2.
    (you can add helper functions as you want).
    """

    def __init__(self):
        AbstractIndexPlayer.__init__(self)
        # TODO: add here if needed

    def get_indices(self, board, value, time_limit) -> (int, int):
        # TODO: erase the following line and implement this function.
        raise NotImplementedError

    # TODO: add here helper functions in class, if needed


# part C
class ABMovePlayer(AbstractMovePlayer):
    """Alpha Beta Move Player,
    implement get_move function according to Alpha Beta MiniMax algorithm
    (you can add helper functions as you want)
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        raise NotImplementedError

    # TODO: add here helper functions in class, if needed


# TODO: use np.countzeros(board)
# part D
class ExpectimaxMovePlayer(AbstractMovePlayer):
    """Expectimax Move Player,
    implement get_move function according to Expectimax algorithm.
    (you can add helper functions as you want)
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)
        self.index_player = ExpectimaxIndexPlayer()

    def get_move(self, board, time_limit) -> Move:
        start_time = time.time()
        optional_moves_score = {}
        max_move_by_depth = Move.UP
        cur_max_value = -1
        cur_depth = 1
        time_limit -= 0.1  # TODO: remove time 0.1
        while time_limit > (time.time() - start_time):
            cur_depth += 1
            for move in Move:
                new_board, done, score = commands[move](board)  # do a run for the board, on trying the "move" direction
                if done:
                    optional_moves_score[move] = self.value(new_board, time_limit - (time.time() - start_time),
                                                            cur_depth, Turn.MOVE_PLAYER_TURN)
            max_move_this_depth = max(optional_moves_score, key=optional_moves_score.get)
            if cur_max_value < optional_moves_score[max_move_this_depth]:
                cur_max_value = optional_moves_score[max_move_this_depth]
                max_move_by_depth = max_move_this_depth
        # print(f'out depth = {cur_depth} max move = {max_move_by_depth}')
        return max_move_by_depth

    def exp_value_state(self, board, time_limit, depth) -> float:
        # TODO: make sure if need to consider index move as a new depth
        start_time = time.time()
        exp_value = 0
        next_states, next_states_probability = self.get_next_index_player_states(board)
        # print(len(next_states))
        # print(np.sum(next_states_probability))
        for i in range(0, len(next_states)):
            exp_value += next_states_probability[i] * self.value(next_states[i],
                                                                 time_limit - (time.time() - start_time), depth - 1,
                                                                 Turn.MOVE_PLAYER_TURN)
        return exp_value

    def max_value_state(self, board, time_limit, depth) -> float:
        start_time = time.time()
        max_val = -1
        for move in Move:
            new_board, done, score = commands[move](board)
            if done:
                max_val = max(max_val, self.value(new_board, time_limit - (time.time() - start_time), depth - 1,
                                                  Turn.INDEX_PLAYER_TURN))
        return max_val

    def value(self, board, time_limit, depth, turn) -> float:
        start_time = time.time()
        if time_limit <= 0 or depth <= 0:  # must decide now, use huristic
            return self.huristic(board)

        if turn == Turn.INDEX_PLAYER_TURN:
            return self.exp_value_state(board, time_limit - (time.time() - start_time), depth)

        else:  # move player turn
            return self.max_value_state(board, time_limit - (time.time() - start_time), depth)

    def huristic(self, board) -> float:
        return self.get_number_of_empty_tiles(board)

    def get_number_of_empty_tiles(self, board) -> int:
        return 16-np.count_nonzero(board)

    # return the first empty tile starting from start_i, start_j (including), if none, then returns -1, -1
    def get_next_empty_tile(self, board, start_i, start_j) -> (int, int):
        if start_i >= len(board):
            start_i = 0
            start_j += 1
        if start_j >= len(board):
            return -1, -1

        for i in range(start_i, len(board)):
            if board[i][start_j] == 0:
                return i, start_j

        for i in range(0, len(board)):
            for j in range(start_j + 1, len(board)):
                if board[i][j] == 0:
                    return i, j
        return -1, -1

    # TODO make sure probability is used right, not for 4
    def get_next_index_player_states(self, board) -> ([], []):
        empty_tiles_count = self.get_number_of_empty_tiles(board)
        probabilities = []
        states_to_return = []
        a, b = self.get_next_empty_tile(board, 0, 0)
        while a != -1:
            board[a][b] = 2
            states_to_return.append(board)
            probabilities.append(PROBABILITY / empty_tiles_count)
            board[a][b] = 4
            states_to_return.append(board)
            probabilities.append((1 - PROBABILITY) / empty_tiles_count)
            board[a][b] = 0
            a, b = self.get_next_empty_tile(board, a + 1, b)
        return states_to_return, probabilities


class ExpectimaxIndexPlayer(AbstractIndexPlayer):  # TODO: not sure of the implementation
    """Expectimax Index Player
    implement get_indices function according to Expectimax algorithm, the value is number between {2,4}.
    (you can add helper functions as you want)
    """

    def __init__(self):
        AbstractIndexPlayer.__init__(self)
        self.random_player = RandomIndexPlayer()

    def get_indices(self, board, value, time_limit) -> (int, int):
        return self.random_player.get_indices(board, value, time_limit)


# Tournament
class ContestMovePlayer(AbstractMovePlayer):
    """Contest Move Player,
    implement get_move function as you want to compete in the Tournament
    (you can add helper functions as you want)
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        raise NotImplementedError

    # TODO: add here helper functions in class, if needed
