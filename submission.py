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

    def get_move(self, board, time_limit) -> Move:
        optional_moves_score = {}
        for move in Move:
            new_board, done, score = commands[move](
                self.copy_board(board))  # do a run for the board, on trying the "move" direction
            if done:
                empty_tiles = max(self.get_number_of_empty_tiles(new_board), 1)
                sum_all = max(self.sum_all_tiles(new_board), 1)
                AMP = math.log10(max(3, sum_all))
                optional_moves_score[move] = self.weighted_score(new_board) / (
                        AMP ** 2) + score + empty_tiles * 10 + 10 * self.aligned(new_board)

        return max(optional_moves_score,
                   key=optional_moves_score.get)  # return comparing to their best value (aka get function)

    def weighted_score(self, board) -> float:
        # matt = [[2, 0.5, 0.5, 2], [0.5, 0.1, 0.1, 0.5],[0.5, 0.1, 0.1, 0.5], [2, 0.5, 0.5, 2]]
        matt = [[2, 1, 1, 1], [0.5, 0.1, 0.1, 0.5], [0.5, 0.1, 0.1, 0.5], [0.5, 0.5, 0.5, 0.5]]
        score = 0
        for i in range(0, len(board)):
            for j in range(0, len(board)):
                score += matt[i][j] * board[i][j]
        return score

    def aligned(self, board):
        sum = 0
        for i in range(0, len(board)):
            for j in range(0, len(board) - 1):
                if board[i][j] == board[i][j + 1]:
                    sum += 1
        for j in range(0, len(board)):
            for i in range(0, len(board) - 1):
                if board[i][j] == board[i + 1][j]:
                    sum += 1
        return sum

    def sum_all_tiles(self, board) -> int:
        to_ret = 0
        for i in range(0, len(board)):
            for j in range(0, len(board)):
                to_ret += board[i][j]
        return to_ret

    def get_number_of_empty_tiles(self, board) -> int:
        return 16 - np.count_nonzero(board)

    def copy_board(self, board):
        new_board = np.array([[0] * 4] * 4)
        for i in range(len(board)):
            for j in range(len(board)):
                new_board[i][j] = board[i][j]
        return new_board


# part B
class MiniMaxMovePlayer(AbstractMovePlayer):
    """MiniMax Move Player,
    implement get_move function according to MiniMax algorithm
    (you can add helper functions as you want).
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)

    def get_move(self, board, time_limit) -> Move:
        time_limit -= 0.1
        start_time = time.time()
        optional_moves_score = {}
        max_move = Move.UP
        cur_max_value = -1
        cur_depth = 0
        while time_limit > (time.time() - start_time):  # iterate over all depth until timeUP
            cur_depth += 1
            for move in Move:
                new_board, done, score = commands[move](
                    board)  # do a run for the board, on trying the "move" direction
                if done:
                    optional_moves_score[move] = self.value(new_board, time_limit - (time.time() - start_time),
                                                            cur_depth, Turn.INDEX_PLAYER_TURN)
                    if move == Move.DOWN:
                        optional_moves_score[move] = 0
                    if move == Move.RIGHT and new_board[0][0] == 0:
                        optional_moves_score[move] = 1
                    if cur_max_value < optional_moves_score[move]:
                        cur_max_value = optional_moves_score[move]
                        max_move = move
                    if time_limit <= (time.time() - start_time):
                        break

        return max_move

    def exp_value_state(self, board, time_limit, depth) -> float:
        start_time = time.time()
        exp_value = np.inf
        next_states, next_states_probability = self.get_next_index_player_states(board)
        for i in range(0, len(next_states)):
            exp_value = min(self.value(next_states[i], time_limit - (time.time() - start_time), depth - 1,
                                       Turn.MOVE_PLAYER_TURN), exp_value)
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

        empty_tiles = max(self.get_number_of_empty_tiles(board), 1)
        sum_all = max(self.sum_all_tiles(board), 1)
        AMP = math.log10(max(3, sum_all))

        weighted_s = self.weighted_score(board) / (AMP ** 2)
        alighned_vval = math.log10(max(self.aligned(board), 2)) * 10
        empty_cells = -math.log2(1 / max(empty_tiles, 2))

        hur = weighted_s + empty_cells + alighned_vval
        if empty_tiles < 5:
            hur = 0.5 * empty_tiles + self.weighted_score(board) / (sum_all ** 2)
        return hur

    def sum_all_tiles(self, board) -> int:
        to_ret = 0
        for i in range(0, len(board)):
            for j in range(0, len(board)):
                to_ret += board[i][j]
        return to_ret

    def weighted_score(self, board) -> float:
        # matt = [[1073741824, 268435456, 67108864, 16777216], [65536, 262144, 1048576, 4194304],[16384, 4096, 1024, 256], [1, 4, 16, 64]]
        # matt = [[64*64, 64*16, 64*4, 64], [64*16, 64*4, 64, 16],[64*4, 64, 16, 4], [64,16, 4, 1]]
        # a,b,c,d = self.get_4_biggest_tiles(board)
        # matt = [[math.log2(a),  math.log2(max(b,2)),math.log2(max(c,2)),  math.log2(max(d,2))], [0.5, 0.1, 0.1, 0.5], [0.5, 0.1, 0.1, 0.5], [0.1, 0.1, 0.1, 0.1]]
        matt = [[2 ** 13, 2 ** 10, 2 ** 8, 5], [0, 0.1, 1, 2], [0.1, 0, 0, 0], [0, 0, 0, 0]]
        score = 0
        for i in range(0, len(board)):
            for j in range(0, len(board)):
                score += matt[i][j] * board[i][j]
        return score

    def get_number_of_empty_tiles(self, board) -> int:
        return 16 - np.count_nonzero(board)

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

    def aligned(self, board):
        sum = 0
        for i in range(0, len(board)):
            for j in range(0, len(board) - 1):
                if board[i][j] == board[i][j + 1]:
                    sum += 1
        for j in range(0, len(board)):
            for i in range(0, len(board) - 1):
                if board[i][j] == board[i + 1][j]:
                    sum += 1
        return sum

        # return all expected next states of the index player, with thier probabilities

    def get_next_index_player_states(self, board, value=None) -> ([], []):
        empty_tiles_count = self.get_number_of_empty_tiles(board)
        probabilities = []
        states_to_return = []
        a, b = self.get_next_empty_tile(board, 0, 0)
        if value is None:
            while a != -1:
                board[a][b] = 2
                states_to_return.append(self.copy_board(board))
                probabilities.append((1 - PROBABILITY) / empty_tiles_count)
                board[a][b] = 4
                states_to_return.append(self.copy_board(board))
                probabilities.append(PROBABILITY / empty_tiles_count)
                board[a][b] = 0
                a, b = self.get_next_empty_tile(board, a + 1, b)
        else:
            while a != -1:
                board[a][b] = value
                states_to_return.append(self.copy_board(board))
                probabilities.append(0)  # irrelevant
                board[a][b] = 0
                a, b = self.get_next_empty_tile(board, a + 1, b)

        return states_to_return, probabilities

    def copy_board(self, board):
        new_board = np.zeros((len(board), len(board)))
        for i in range(len(board)):
            for j in range(len(board)):
                new_board[i][j] = board[i][j]
        return new_board


class MiniMaxIndexPlayer(AbstractIndexPlayer):
    """MiniMax Index Player,
    this player is the opponent of the move player and need to return the indices on the board where to put 2.
    the goal of the player is to reduce move player score.
    implement get_indices function according to MiniMax algorithm, the value in minimax player value is only 2.
    (you can add helper functions as you want).
    """

    def __init__(self):
        AbstractIndexPlayer.__init__(self)
        self.max_agent = MiniMaxMovePlayer()

    def get_indices(self, board, value, time_limit) -> (int, int):
        start_time = time.time() - 0.1
        min_val = np.inf
        min_board = None
        depth = 0
        next_states, _ = self.max_agent.get_next_index_player_states(board, value)
        while time_limit - (time.time() - start_time) > 0:
            depth += 1

            for state in next_states:
                tmp = self.max_agent.exp_value_state(state, time_limit - (time.time() - start_time), depth)
                if tmp < min_val:
                    min_val = tmp
                    min_board = state
                if time_limit <= (time.time() - start_time):
                    break
        # time UP
        try:
            return self.find_diff_indices(board, min_board)
        except:
            return self.first_empty_tile(board)

    def first_empty_tile(self, board):
        for i in range(0, len(board)):
            for j in range(0, len(board)):
                if board[i][j] == 0:
                    return i, j

    def find_diff_indices(self, board, min_board):
        for i in range(0, len(board)):
            for j in range(0, len(board)):
                if board[i][j] != min_board[i][j]:
                    return i, j

        return -1, -1


# part C
class ABMovePlayer(AbstractMovePlayer):
    """Alpha Beta Move Player,
    implement get_move function according to Alpha Beta MiniMax algorithm
    (you can add helper functions as you want)
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)
        self.alpha = float('-inf')
        self.beta = float('inf')

    def get_move(self, board, time_limit) -> Move:
        time_limit -= 0.1

        start_time = time.time()
        max_move = Move.UP
        cur_max_value = -np.inf
        cur_depth = 1
        while time_limit > (time.time() - start_time):  # iterate over all depth until timeUP
            cur_depth += 1
            for move in Move:
                new_board, done, score = commands[move](board)  # do a run for the board, on trying the "move" direction
                if done:
                    # reset last AB

                    alpha = float('-inf')
                    beta = float('inf')
                    v = self.ABsearch(new_board, time_limit, start_time, 0, cur_depth, alpha, beta)
                    if cur_max_value < v:
                        cur_max_value = v
                        max_move = move
                    if time_limit <= (time.time() - start_time):
                        break
        return max_move

    # 1 = max, 0 = min

    def ABsearch(self, board, time_limit, start_time, agent, depth, alpha, beta) -> float:

        # debug only
        # self.alpha = float('-inf')
        # self.beta = float('inf')
        #########
        if depth == 0 or time.time() - start_time >= time_limit:
            return self.huristic(board)
        if agent == 1:
            curr_max = float('-inf')
            for move in Move:
                new_board, done, score = commands[move](board)  # do a run for the board, on trying the "move" direction
                if done:
                    v = self.ABsearch(new_board, time_limit, start_time, 0, depth - 1, alpha, beta)
                    curr_max = max(v, curr_max)
                    alpha = max(curr_max, alpha)
                    if curr_max >= beta:
                        return self.huristic(board)
                    if time.time() - start_time >= time_limit:
                        break
            return curr_max

        elif agent == 0:
            curr_min = float('inf')
            states, _ = self.get_next_index_player_states(self.copy_board(board), value=2)
            for state in states:
                v = self.ABsearch(self.copy_board(state), time_limit, start_time, 1, depth - 1, alpha, beta)
                # if v is None:
                #    v = self.ABsearch(state, time_limit, start_time, 1, depth - 1, alpha, beta)
                curr_min = min(v, curr_min)
                beta = min(curr_min, beta)
                if curr_min <= alpha:
                    # print('prune')
                    return self.huristic(board)
                if time.time() - start_time >= time_limit:
                    break
            return curr_min

    def huristic(self, board) -> float:

        empty_tiles = max(self.get_number_of_empty_tiles(board), 1)
        sum_all = max(self.sum_all_tiles(board), 1)
        AMP = math.log10(max(3, sum_all))

        weighted_s = self.weighted_score(board) / (AMP ** 2)
        alighned_vval = math.log10(max(self.aligned(board), 2)) * 10
        empty_cells = -math.log2(1 / max(empty_tiles, 2))

        hur = weighted_s + empty_cells + alighned_vval
        if empty_tiles < 5:
            hur = 0.5 * empty_tiles + self.weighted_score(board) / (sum_all ** 2)
        return hur

    def sum_all_tiles(self, board) -> int:
        to_ret = 0
        for i in range(0, len(board)):
            for j in range(0, len(board)):
                to_ret += board[i][j]
        return to_ret

    def weighted_score(self, board) -> float:
        # matt = [[1073741824, 268435456, 67108864, 16777216], [65536, 262144, 1048576, 4194304],[16384, 4096, 1024, 256], [1, 4, 16, 64]]
        # matt = [[64*64, 64*16, 64*4, 64], [64*16, 64*4, 64, 16],[64*4, 64, 16, 4], [64,16, 4, 1]]
        # a,b,c,d = self.get_4_biggest_tiles(board)
        # matt = [[math.log2(a),  math.log2(max(b,2)),math.log2(max(c,2)),  math.log2(max(d,2))], [0.5, 0.1, 0.1, 0.5], [0.5, 0.1, 0.1, 0.5], [0.1, 0.1, 0.1, 0.1]]
        matt = [[2 ** 13, 2 ** 10, 2 ** 8, 5], [0, 0.1, 1, 2], [0.1, 0, 0, 0], [0, 0, 0, 0]]
        score = 0
        for i in range(0, len(board)):
            for j in range(0, len(board)):
                score += matt[i][j] * board[i][j]
        return score

    def get_number_of_empty_tiles(self, board) -> int:
        return len(board) ** 2 - np.count_nonzero(board)

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

    def aligned(self, board):
        sum = 0
        for i in range(0, len(board)):
            for j in range(0, len(board) - 1):
                if board[i][j] == board[i][j + 1]:
                    sum += 1
        for j in range(0, len(board)):
            for i in range(0, len(board) - 1):
                if board[i][j] == board[i + 1][j]:
                    sum += 1
        return sum

        # return all expected next states of the index player, with thier probabilities

    def get_next_index_player_states(self, board, value=None) -> ([], []):
        empty_tiles_count = self.get_number_of_empty_tiles(board)
        probabilities = []
        states_to_return = []
        a, b = self.get_next_empty_tile(board, 0, 0)
        if value is None:
            while a != -1:
                board[a][b] = 2
                states_to_return.append(self.copy_board(board))
                probabilities.append((1 - PROBABILITY) / empty_tiles_count)
                board[a][b] = 4
                states_to_return.append(self.copy_board(board))
                probabilities.append(PROBABILITY / empty_tiles_count)
                board[a][b] = 0
                a, b = self.get_next_empty_tile(board, a + 1, b)
        else:
            while a != -1:
                board[a][b] = value
                states_to_return.append(self.copy_board(board))
                probabilities.append(0)  # irrelevant
                board[a][b] = 0
                a, b = self.get_next_empty_tile(board, a + 1, b)

        return states_to_return, probabilities

    def copy_board(self, board):
        new_board = np.zeros((len(board), len(board)))
        for i in range(len(board)):
            for j in range(len(board)):
                new_board[i][j] = board[i][j]
        return new_board


# part D
class ExpectimaxMovePlayer(AbstractMovePlayer):
    """Expectimax Move Player,
    implement get_move function according to Expectimax algorithm.
    (you can add helper functions as you want)
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)

    def get_move(self, board, time_limit) -> Move:
        time_limit -= 0.1
        start_time = time.time()
        optional_moves_score = {}
        max_move = Move.UP
        cur_max_value = -1
        cur_depth = 0
        while time_limit > (time.time() - start_time):  # iterate over all depth until timeUP
            cur_depth += 1
            for move in Move:
                new_board, done, score = commands[move](board)  # do a run for the board, on trying the "move" direction
                if done:
                    optional_moves_score[move] = self.value(new_board, time_limit - (time.time() - start_time),
                                                            cur_depth, Turn.INDEX_PLAYER_TURN)
                    if move == Move.DOWN:
                        optional_moves_score[move] = 0
                    if move == Move.RIGHT and new_board[0][0] == 0:
                        optional_moves_score[move] = 1
                    if cur_max_value < optional_moves_score[move]:
                        cur_max_value = optional_moves_score[move]
                        max_move = move
                    if time_limit <= (time.time() - start_time):
                        break

        return max_move

    def exp_value_state(self, board, time_limit, depth) -> float:
        start_time = time.time()
        exp_value = 0
        next_states, next_states_probability = self.get_next_index_player_states(board)
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

        empty_tiles = max(self.get_number_of_empty_tiles(board), 1)
        sum_all = max(self.sum_all_tiles(board), 1)
        AMP = math.log10(max(3, sum_all))

        weighted_s = self.weighted_score(board) / (AMP ** 2)
        alighned_vval = math.log10(max(self.aligned(board), 2)) * 10
        empty_cells = -math.log2(1 / max(empty_tiles, 2))

        hur = weighted_s + empty_cells + alighned_vval
        if empty_tiles < 3:
            hur = 0.5 * empty_tiles
        return hur

    def sum_all_tiles(self, board) -> int:
        to_ret = 0
        for i in range(0, len(board)):
            for j in range(0, len(board)):
                to_ret += board[i][j]
        return to_ret

    def weighted_score(self, board) -> float:
        # matt = [[1073741824, 268435456, 67108864, 16777216], [65536, 262144, 1048576, 4194304],[16384, 4096, 1024, 256], [1, 4, 16, 64]]
        # matt = [[64*64, 64*16, 64*4, 64], [64*16, 64*4, 64, 16],[64*4, 64, 16, 4], [64,16, 4, 1]]
        # a,b,c,d = self.get_4_biggest_tiles(board)
        # matt = [[math.log2(a),  math.log2(max(b,2)),math.log2(max(c,2)),  math.log2(max(d,2))], [0.5, 0.1, 0.1, 0.5], [0.5, 0.1, 0.1, 0.5], [0.1, 0.1, 0.1, 0.1]]
        matt = [[2 ** 13, 2 ** 10, 2 ** 8, 5], [0, 0.1, 1, 2], [0.1, 0, 0, 0], [0, 0, 0, 0]]
        # matt = [[2 ** 13, 2 ** 10, 2 ** 8, 5], [0, 0.1, 0.5, 0.6], [0.1, 0, 0, 0], [0, 0, 0, 0]]
        score = 0
        for i in range(0, len(board)):
            for j in range(0, len(board)):
                score += matt[i][j] * board[i][j]
        return score

    def get_number_of_empty_tiles(self, board) -> int:
        return 16 - np.count_nonzero(board)

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

    def aligned(self, board):
        sum = 0
        for i in range(0, len(board)):
            for j in range(0, len(board) - 1):
                if board[i][j] == board[i][j + 1]:
                    sum += 1
        for j in range(0, len(board)):
            for i in range(0, len(board) - 1):
                if board[i][j] == board[i + 1][j]:
                    sum += 1
        return sum

    # return all expected next states of the index player, with thier probabilities
    def get_next_index_player_states(self, board, value=None) -> ([], []):
        empty_tiles_count = self.get_number_of_empty_tiles(board)
        probabilities = []
        states_to_return = []
        a, b = self.get_next_empty_tile(board, 0, 0)
        if value is None:
            while a != -1:
                board[a][b] = 2
                states_to_return.append(self.copy_board(board))
                probabilities.append((1 - PROBABILITY) / empty_tiles_count)
                board[a][b] = 4
                states_to_return.append(self.copy_board(board))
                probabilities.append(PROBABILITY / empty_tiles_count)
                board[a][b] = 0
                a, b = self.get_next_empty_tile(board, a + 1, b)
        else:
            while a != -1:
                board[a][b] = value
                states_to_return.append(self.copy_board(board))
                probabilities.append(0)  # irrelevant
                board[a][b] = 0
                a, b = self.get_next_empty_tile(board, a + 1, b)

        return states_to_return, probabilities

    def copy_board(self, board):
        new_board = np.zeros((len(board), len(board)))
        for i in range(len(board)):
            for j in range(len(board)):
                new_board[i][j] = board[i][j]
        return new_board


class ExpectimaxIndexPlayer(AbstractIndexPlayer):
    """Expectimax Index Player
    implement get_indices function according to Expectimax algorithm, the value is number between {2,4}.
    (you can add helper functions as you want)
    """

    def __init__(self):
        AbstractIndexPlayer.__init__(self)
        self.max_agent = ExpectimaxMovePlayer()

    def get_indices(self, board, value, time_limit) -> (int, int):
        start_time = time.time() - 0.1
        min_val = np.inf
        min_board = None
        depth = 0
        next_states, _ = self.max_agent.get_next_index_player_states(board, value)
        while time_limit - (time.time() - start_time) > 0:
            depth += 1

            for state in next_states:
                tmp = self.max_agent.exp_value_state(state, time_limit - (time.time() - start_time), depth)
                if tmp < min_val:
                    min_val = tmp
                    min_board = state
                if time_limit <= (time.time() - start_time):
                    break
        # time UP
        return self.find_diff_indices(board, min_board)

    def find_diff_indices(self, board, min_board):
        for i in range(0, len(board)):
            for j in range(0, len(board)):
                if board[i][j] != min_board[i][j]:
                    return i, j

        return -1, -1


# Tournament
class ContestMovePlayer(AbstractMovePlayer):
    """Contest Move Player,
    implement get_move function as you want to compete in the Tournament
    (you can add helper functions as you want)
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)

    def get_move(self, board, time_limit) -> Move:
        time_limit -= 0.1
        start_time = time.time()
        optional_moves_score = {}
        max_move = Move.UP
        cur_max_value = -1
        cur_depth = 0
        while time_limit > (time.time() - start_time):  # iterate over all depth until timeUP
            cur_depth += 1
            for move in Move:
                new_board, done, score = commands[move](board)  # do a run for the board, on trying the "move" direction
                if done:
                    optional_moves_score[move] = self.value(new_board, time_limit - (time.time() - start_time),
                                                            cur_depth, Turn.INDEX_PLAYER_TURN)
                    if move == Move.DOWN:
                        optional_moves_score[move] = 0
                    if move == Move.RIGHT and new_board[0][0] == 0:
                        optional_moves_score[move] = 1
                    if cur_max_value < optional_moves_score[move]:
                        cur_max_value = optional_moves_score[move]
                        max_move = move
                    if time_limit <= (time.time() - start_time):
                        break

        return max_move

    def exp_value_state(self, board, time_limit, depth) -> float:
        start_time = time.time()
        exp_value = 0
        next_states, next_states_probability = self.get_next_index_player_states(board)
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

        empty_tiles = max(self.get_number_of_empty_tiles(board), 1)
        sum_all = max(self.sum_all_tiles(board), 1)
        AMP = math.log10(max(3, sum_all))

        weighted_s = self.weighted_score(board) / (AMP ** 2)
        alighned_vval = math.log10(max(self.aligned(board), 2)) * 10
        empty_cells = -math.log2(1 / max(empty_tiles, 2))

        hur = weighted_s + empty_cells + alighned_vval
        if empty_tiles < 3:
            hur = 0.5 * empty_tiles
        return hur

    def sum_all_tiles(self, board) -> int:
        to_ret = 0
        for i in range(0, len(board)):
            for j in range(0, len(board)):
                to_ret += board[i][j]
        return to_ret

    def weighted_score(self, board) -> float:
        # matt = [[1073741824, 268435456, 67108864, 16777216], [65536, 262144, 1048576, 4194304],[16384, 4096, 1024, 256], [1, 4, 16, 64]]
        # matt = [[64*64, 64*16, 64*4, 64], [64*16, 64*4, 64, 16],[64*4, 64, 16, 4], [64,16, 4, 1]]
        # a,b,c,d = self.get_4_biggest_tiles(board)
        # matt = [[math.log2(a),  math.log2(max(b,2)),math.log2(max(c,2)),  math.log2(max(d,2))], [0.5, 0.1, 0.1, 0.5], [0.5, 0.1, 0.1, 0.5], [0.1, 0.1, 0.1, 0.1]]
        matt = [[2 ** 13, 2 ** 10, 2 ** 8, 5], [0.1, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0]]
        # matt = [[2 ** 13, 2 ** 10, 2 ** 8, 5], [0, 0.1, 0.5, 0.6], [0.1, 0, 0, 0], [0, 0, 0, 0]]
        score = 0
        for i in range(0, len(board)):
            for j in range(0, len(board)):
                score += matt[i][j] * board[i][j]
        return score

    def get_number_of_empty_tiles(self, board) -> int:
        return 16 - np.count_nonzero(board)

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

    def aligned(self, board):
        sum = 0
        for i in range(0, len(board)):
            for j in range(0, len(board) - 1):
                if board[i][j] == board[i][j + 1]:
                    sum += 1
        for j in range(0, len(board)):
            for i in range(0, len(board) - 1):
                if board[i][j] == board[i + 1][j]:
                    sum += 1
        return sum

    # return all expected next states of the index player, with thier probabilities
    def get_next_index_player_states(self, board, value=None) -> ([], []):
        empty_tiles_count = self.get_number_of_empty_tiles(board)
        probabilities = []
        states_to_return = []
        a, b = self.get_next_empty_tile(board, 0, 0)
        if value is None:
            while a != -1:
                board[a][b] = 2
                states_to_return.append(self.copy_board(board))
                probabilities.append((1 - PROBABILITY) / empty_tiles_count)
                board[a][b] = 4
                states_to_return.append(self.copy_board(board))
                probabilities.append(PROBABILITY / empty_tiles_count)
                board[a][b] = 0
                a, b = self.get_next_empty_tile(board, a + 1, b)
        else:
            while a != -1:
                board[a][b] = value
                states_to_return.append(self.copy_board(board))
                probabilities.append(0)  # irrelevant
                board[a][b] = 0
                a, b = self.get_next_empty_tile(board, a + 1, b)

        return states_to_return, probabilities

    def copy_board(self, board):
        new_board = np.zeros((len(board), len(board)))
        for i in range(len(board)):
            for j in range(len(board)):
                new_board[i][j] = board[i][j]
        return new_board
