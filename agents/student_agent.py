# Student agent: Add your own agent here
# import random
from agents.agent import Agent
from store import register_agent
import numpy as np


def check_endgame(chess_board, p0_pos, p1_pos):
    """
    This function check if the game ends and compute the current score of the agents.

    Inputs:
    chess_board: numpy array (indicate the game state, the place of walls)
    p0_pos: tuple (the position of player 0)
    p1_pos: tuple (the position of player 1)

    Returns:
    is_endgame : bool (Whether the game ends)
    player_win : int (the winner of the game)
        1 - when player 0 wins
        -1 - when player 1 wins
        0 - when tie happens
        otherwise - when game is not end
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
    # search the available path
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

    # find all of the cells the player 0/1 can reach(without max step)
    p0_r = find(tuple(p0_pos))
    p1_r = find(tuple(p1_pos))
    # compute the score of each player
    p0_score = list(father.values()).count(p0_r)
    p1_score = list(father.values()).count(p1_r)
    # check if game end or not
    if p0_r == p1_r:
        return False, p0_score
    # find the winner of the game
    if p0_score > p1_score:
        player_win = 1
    elif p0_score < p1_score:
        player_win = -1
    else:
        player_win = 0
    return True, player_win


def check_valid_step(chess_board, adv_pos, max_step, start_pos, end_pos):
    """
    This function check if the step the agent takes is valid (reachable and within max steps).

    Inputs:
    chess_board: numpy array (indicate the game state, the place of walls)
    adv_pos: tuple (the position of adversary)
    max_step: int (the max step agent can move)
    start_pos : tuple (the start position of the agent)
    end_pos : tuple (the end position of the agent)

    Returns:
    is_reached: bool (indicate whether agent can move from start_pos to end_pos within max step)
    """
    # if the end point is exactly the start point
    if start_pos == end_pos:
        return True
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    # implement the breath first search to find a path from start_pos to end_pos
    state_queue = [(start_pos, 0)]
    # record the visited cells
    visited = {tuple(start_pos)}
    # indicated whether the end_pos is reached
    is_reached = False
    while state_queue and not is_reached:
        cur_pos, cur_step = state_queue.pop(0)
        r, c = cur_pos
        # if the max_step limit is reached
        if cur_step == max_step:
            break
        # move from start_pos to next available position
        for dir, move in enumerate(moves):
            if chess_board[r, c, dir]:
                continue
            next_pos = tuple(map(sum, zip(cur_pos, move)))
            # if the next position has adversary or has visited before
            if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                continue
            # if reach the end_pos
            if np.array_equal(next_pos, end_pos):
                is_reached = True
                break
            # add next available positions to the visited list
            visited.add(tuple(next_pos))
            state_queue.append((next_pos, cur_step + 1))

    return is_reached


def compute_weight(chess_board, cur_pos, adv_pos):
    """
    This function compute the weight(heuristic) and the available directions
    which can place a wall.

    Inputs:
    chess_board: numpy array (indicate the game state, the place of walls)
    cur_pos : tuple (the next available position the agent can reach)
    adv_pos: tuple (the position of adversary)

    Returns:
    weight: int (the weight of the path from the current to the next position - heuristic value)
    available_direction: list of int (the directions which can place a wall)
    """
    # extract the row and column value for current position and adversary
    r, c = cur_pos
    r1, c1 = adv_pos
    # record the number of wall already on the current position
    wall = 0
    # record the directions which can place a wall
    available_direction = []
    # calculate the Manhattan Distance between the position of current step and adversary
    distance = abs(r - r1) + abs(c - c1)
    # for all directions
    for i in range(4):
        # if there is a wall
        if chess_board[r][c][i]:
            wall = wall + 1
        # otherwise
        else:
            available_direction.append(i)
    # compute the initial weight
    weight = wall + distance
    # if there is already three walls on the current position
    # agent will lose the game if reaches the current position
    if wall == 3:
        # increase the weight to reach the current position
        weight = 100
    # the position will less walls will preferred
    # modify the weight considering to the number of walls in the current position
    if wall == 2:
        weight = weight * 3
    if wall == 1:
        weight = weight * 2
    return weight, available_direction


def find_valid_step(board_size, my_pos, max_step):
    """
    This function find the valid steps the agent can reach from current position within max step
    without considering the walls on the board for now
    Inputs:
    board_size: int (indicate the size of the chess board)
    my_pos : tuple (the current position of the agent)
    max_step: int (the maximum step the agent can move)

    Returns:
    valid_step: list of tuple (the valid steps the agent can reach)
    """
    # extract the row and column value for current position
    r, c = my_pos
    # record the valid steps
    valid_step = []
    # find the cells agent can reach from the current position within max step
    for i in range(max(0, r - max_step), min(r + max_step + 1, board_size)):
        for j in range(max(0, c - max_step), min(c + max_step + 1, board_size)):
            if abs(r - i) + abs(c - j) <= max_step:
                valid_step.append((i, j))
    return valid_step


def compute_direction(chess_board, cur_pos):
    """
    This function compute the available directions which can place a wall.

    Inputs:
    chess_board: numpy array (indicate the game state, the place of walls)
    cur_pos : tuple (the next available position the agent can reach)

    Returns:
    available_direction: list of int (the directions which can place a wall)
    """
    r2, c2 = cur_pos
    available_direction2 = []
    for i in range(4):
        if not chess_board[r2][c2][i]:
            available_direction2.append(i)
    return available_direction2


def find_best_step(chess_board, my_pos, adv_pos, max_step):
    """
    This function find the best next step for agent.

    Inputs:
    chess_board: numpy array (indicate the game state, the place of walls)
    my_pos : tuple (the current position the agent)
    adv_pos: tuple (the position of adversary)
    max_step: int (the maximum step the agent can move)

    Returns:
    best_step: tuple of ((x, y), dir)
    """
    # compute the size of the chess board
    board_size = chess_board.shape[0]
    # find the valid steps the agent can reach from current position within max step
    # without considering the walls on the board for now
    next_list = find_valid_step(board_size, my_pos, max_step)
    # initialize the weight and best_step
    weight = 10000
    best_step = (my_pos, compute_direction(chess_board, my_pos)[0])
    # used for compute opposite barrier
    opp = ((-1, 0), (0, 1), (1, 0), (0, -1))
    opposites = {0: 2, 1: 3, 2: 0, 3: 1}
    # for all valid steps
    for move in next_list:
        # check if agent can reach within max step
        if check_valid_step(chess_board, adv_pos, max_step, my_pos, move):
            # compute the weight and available_direction for next steo
            weight1, available_direction = compute_weight(chess_board, move, adv_pos)
            # for all available directions
            for i in available_direction:
                r, c = move
                opps = opp[i]
                # set the current barrier and the opposite barrier to True
                chess_board[r][c][i] = True
                chess_board[r + opps[0], c + opps[1], opposites[i]] = True
                # check if the game will end once the next step is chosen
                end_game, winner = check_endgame(chess_board, move, adv_pos)
                # if the game is end
                if end_game:
                    # if the agent is the winner
                    if winner == 1:
                        # return this step as the best step
                        return move, i
                    # if adversary is the winner
                    elif winner == -1:
                        # remove this direction from available_direction
                        available_direction.remove(i)
                        # increase the weight such that other position is preferred then this
                        if move == my_pos:
                            weight1 = 10000
                # set the current barrier and the opposite barrier to False
                chess_board[r][c][i] = False
                chess_board[r + opps[0], c + opps[1], opposites[i]] = False
            # if there does not exist a good direction to move
            if available_direction == []:
                continue
            # if this step is better than the best step assigned before
            if weight1 < weight:
                weight = weight1
                best_step = (move, available_direction[0])

    return best_step


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.state = None

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

        Please check the sample implementation in agents/random_agent.py or
        agents/human_agent.py for more details.
        """
        my_pos, direction = find_best_step(chess_board, my_pos, adv_pos, max_step)

        # update choice of direction since we prefer split the space between my and avd
        r,c = my_pos
        avaliable_dir = compute_direction(chess_board, my_pos)

        for dir in avaliable_dir:
            if dir == 0 and (r-1,c) == adv_pos:        # up
                direction = 0
            elif dir == 1 and (r,c+1) == adv_pos:        # right
                direction = 1
            elif dir == 2 and (r+1,c) == adv_pos:        # left
                direction = 2
            elif dir == 3 and (r,c-1) == adv_pos:        # down
                direction = 3

        return my_pos, direction