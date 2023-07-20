# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import copy
import numpy as np


def check_endgame(chess_board, p0_pos, p1_pos):
    """
    Check if the game ends and compute the current score of the agents.

    Returns
    -------
    is_endgame : bool
        Whether the game ends.
    player_1_score : int
        The score of player 1.
    player_2_score : int
        The score of player 2.
    """
    board_size = chess_board.shape[0]
    father = dict()
    for r in range(board_size):
        for c in range(board_size):
            father[(r, c)] = (r, c)

    def find(pos):
        if father[pos] != pos:
            father[pos] = find(father[pos])
        return father[pos]

    def union(pos1, pos2):
        father[pos1] = pos2

    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

    for r in range(board_size):
        for c in range(board_size):
            for dir, move in enumerate(
                    moves[1:3]
            ):  # Only check down and right
                if chess_board[r, c, dir + 1]:
                    continue
                pos_a = find((r, c))
                pos_b = find((r + move[0], c + move[1]))
                if pos_a != pos_b:
                    union(pos_a, pos_b)

    for r in range(board_size):
        for c in range(board_size):
            find((r, c))
    p0_r = find(tuple(p0_pos))
    p1_r = find(tuple(p1_pos))
    p0_score = list(father.values()).count(p0_r)
    p1_score = list(father.values()).count(p1_r)
    if p0_r == p1_r:
        return False, p0_score
    if p0_score > p1_score:
        player_win = 1
    elif p0_score < p1_score:
        player_win = -1
    else:
        player_win = 0  # Tie
    return True, player_win


def check_valid_step(chess_board, adv_pos, max_step, start_pos, end_pos):
    """
    Check if the step the agent takes is valid (reachable and within max steps).

    Parameters
    ----------
    start_pos : tuple
        The start position of the agent.
    end_pos : np.ndarray
        The end position of the agent.
    barrier_dir : int
        The direction of the barrier.
    """
    # Endpoint already has barrier or is boarder
    if np.array_equal(start_pos, end_pos):
        return True
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    # BFS
    state_queue = [(start_pos, 0)]
    visited = {tuple(start_pos)}
    is_reached = False
    while state_queue and not is_reached:
        cur_pos, cur_step = state_queue.pop(0)
        r, c = cur_pos
        if cur_step == max_step:
            break

        for dir, move in enumerate(moves):
            if chess_board[r, c, dir]:
                continue
            next_pos = tuple(map(sum, zip(cur_pos, move)))

            if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                continue
            if np.array_equal(next_pos, end_pos):
                is_reached = True
                break

            visited.add(tuple(next_pos))
            state_queue.append((next_pos, cur_step + 1))

    return is_reached


def set_barrier(chess_board, r, c, d):
    # Set the barrier to True
    chess_board[r, c, d] = True
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    opposites = {0: 2, 1: 3, 2: 0, 3: 1}
    # Set the opposite barrier to True
    move = moves[d]
    chess_board[r + move[0], c + move[1], opposites[d]] = True
    return chess_board


def remove_barrier(chess_board, r, c, d):
    # Set the barrier to True
    chess_board[r, c, d] = False
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    opposites = {0: 2, 1: 3, 2: 0, 3: 1}
    # Set the opposite barrier to True
    move = moves[d]
    chess_board[r + move[0], c + move[1], opposites[d]] = False
    return chess_board


def compute_weight_dir(chess_board, cur_pos, adv_pos):
    r, c = cur_pos
    r1, c1 = adv_pos
    wall = 0
    available_direction = []
    distance = abs(r - r1) + abs(c - c1)
    for i in range(4):
        if chess_board[r][c][i]:
            wall = wall + 1
        else:
            available_direction.append(i)
    weight = wall + distance*2
    if wall == 3:
        weight = 1000
    if wall == 2:
        weight = weight * 3
    if wall == 1:
        weight = weight * 2
    return weight, available_direction


def compute_dir(chess_board, cur_pos):
    r2, c2 = cur_pos
    available_direction2 = []
    for i in range(4):
        if not chess_board[r2][c2][i]:
            available_direction2.append(i)
    return available_direction2


def find_valid_step(board_size, my_pos, max_step):
    r, c = my_pos
    valid_step = []
    for i in range(max(0, r - max_step), min(r + max_step + 1, board_size)):
        for j in range(max(0, c - max_step), min(c + max_step + 1, board_size)):
            if abs(r - i) + abs(c - j) <= max_step:
                valid_step.append((i, j))
    return valid_step


def predict_adv_step(chess_board, available_direction1, step, move):
    r1, c1 = step
    for j in available_direction1:
        chess_board = set_barrier(chess_board, r1, c1, j)
        end_game, winner = check_endgame(chess_board, step, move)
        if end_game and (winner == 1):
            chess_board = remove_barrier(chess_board, r1, c1, j)
            return True
        chess_board = remove_barrier(chess_board, r1, c1, j)
    return False


@register_agent("student_agent3")
class StudentAgent3(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent3, self).__init__()
        self.name = "StudentAgent3"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.state = None
        self.k = 0

    def find_best_step(self, chess_board, my_pos, adv_pos, max_step):
        self.k = self.k + 1
        board_size = chess_board.shape[0]
        next_list = find_valid_step(board_size, my_pos, max_step)
        weight = 10000
        best_step = (my_pos, compute_dir(chess_board, my_pos)[0])
        adv_step = find_valid_step(board_size, adv_pos, max_step)
        bad_initial = False
        q = True
        for move in next_list:
            r, c = move
            if check_valid_step(chess_board, adv_pos, max_step, my_pos, move):
                weight1, available_direction = compute_weight_dir(chess_board, move, adv_pos)
                temp_dir = copy.deepcopy(available_direction)
                #print(move)
                for i in available_direction:
                    #print(i)
                    chess_board = set_barrier(chess_board, r, c, i)
                    end_game, winner = check_endgame(chess_board, move, adv_pos)
                    if end_game:
                        if winner == 1:
                            #print(222)
                            best_step = (move, i)
                            return best_step
                        elif winner == -1:
                            #print(111)
                            if move == my_pos:
                                bad_initial = True
                            temp_dir.remove(i)
                    elif self.k >= (board_size / 2):
                        #print(333)
                        for step in adv_step:
                            if check_valid_step(chess_board, move, max_step, adv_pos, step):
                                available_direction1 = compute_dir(chess_board, step)
                                if predict_adv_step(chess_board, available_direction1, step, move):
                                    #print(666)
                                    temp_dir.remove(i)
                                    if move == my_pos:
                                        bad_initial = True
                                    q = False
                                    break
                        if q:
                            #print(555)
                            best_step = (move, available_direction[0])
                            return best_step

                    chess_board = remove_barrier(chess_board, r, c, i)
                    available_direction = copy.deepcopy(temp_dir)
                if not available_direction:
                    continue
                elif bad_initial and (move == my_pos):
                    bad_initial = False
                    continue
                if weight1 < weight:
                    weight = weight1
                    # print(move)
                    # print(available_direction)
                    best_step = (move, available_direction[0])
        #print(best_step)
        return best_step

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        my_pos, dic = self.find_best_step(chess_board, my_pos, adv_pos, max_step)
        return my_pos, dic