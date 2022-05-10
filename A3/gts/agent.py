"""
An AI player for Othello.
"""

import random
import sys
import time

# You can use the functions in othello_shared to write your AI
from othello_shared import find_lines, get_possible_moves, get_score, play_move

# a global dictionary to caching seen states, maps board states to their minimax value
caching_states = {}

def eprint(*args, **kwargs): #you can use this for debugging, as it will print to sterr and not stdout
    print(*args, file=sys.stderr, **kwargs)

# Method to compute utility value of terminal state
def compute_utility(board, color):
    n_dark, n_light = get_score(board)
    if color == 1:
        return n_dark - n_light
    else:
        return n_light - n_dark

# evaluate the score by stable disks that cannot be flipped at corner and along edge
# corner is the most important, with its subsequent edge disks being stable as well
def stable_score(board, color, scores):
    n = len(board)

    # left corner and edge
    if board[0][0] == color:
        scores[color-1] += 3
        for i in range(1, n):
            if board[0][i] == color:
                scores[color-1] += 1
            else:
                break
        for i in range(1, n):
            if board[i][0] == color:
                scores[color-1] += 1
            else:
                break

    if board[0][n-1] == color:
        scores[color-1] += 3
        for i in range(1, n):
            if board[0][n-1-i] == color:
                scores[color-1] += 1
            else:
                break
        for i in range(1, n):
            if board[i][n-1] == color:
                scores[color-1] += 1
            else:
                break

    if board[n-1][0] == color:
        scores[color-1] += 3
        for i in range(1, n):
            if board[n-1-i][0] == color:
                scores[color-1] += 1
            else:
                break
        for i in range(1, n):
            if board[n-1][i] == color:
                scores[color-1] += 1
            else:
                break

    if board[n-1][n-1] == color:
        scores[color-1] += 3
        for i in range(1, n):
            if board[n-1-i][n-1] == color:
                scores[color-1] += 1
            else:
                break
        for i in range(1, n):
            if board[n-1][n-1-i] == color:
                scores[color-1] += 1
            else:
                break
    return scores

# penalize for X-square and isolated C-square
def XC_square(board, color, scores, n_moves):
    n = len(board)

    # penalize more for opening
    if n_moves < (n * n / 6):
        penalty = 2
    else:
        penalty = 1

    if board[0][0] == 0:
        # X-square
        if board[1][1] == color:
            scores[color-1] -= penalty

        # isolated C-square
        if board[0][1] == color:
            count = 0
            for i in range(2, n):
                if board[0][i] == color:
                    count += 1
                else:
                    break
            if count < n - 1:
                scores[color-1] -= penalty
        # isolated C-square
        if board[1][0] == color:
            count = 0
            for i in range(2, n):
                if board[i][0] == color:
                    count += 1
                else:
                    break
            if count < n - 1:
                scores[color-1] -= penalty

    if board[0][n-1] == 0:
        if board[1][n-2] == color:
            scores[color-1] -= penalty
        if board[0][n-2] == color:
            count = 0
            for i in range(2, n):
                if board[0][n-1-i] == color:
                    count += 1
                else:
                    break
            if count < n - 1:
                scores[color-1] -= penalty

        if board[1][n-1] == color:
            count = 0
            for i in range(2, n):
                if board[i][n-1] == color:
                    count += 1
                else:
                    break
            if count < n - 1:
                scores[color-1] -= penalty

    if board[n-1][0] == 0:
        if board[n-2][1] == color:
            scores[color-1] -= penalty
        if board[n-2][0] == color:
            count = 0
            for i in range(2, n):
                if board[n-i-1][0] == color:
                    count += 1
                else:
                    break
            if count < n - 1:
                scores[color-1] -= penalty

        if board[n-1][1] == color:
            count = 0
            for i in range(2, n):
                if board[n-1][i] == color:
                    count += 1
                else:
                    break
            if count < n - 1:
                scores[color-1] -= penalty

    if board[n-1][n-1] == 0:
        if board[n-2][n-2] == color:
            scores[color-1] -= penalty
        if board[n-1][n-2] == color:
            count = 0
            for i in range(2, n):
                if board[n-1][n-1-i] == color:
                    count += 1
                else:
                    break
            if count < n - 1:
                scores[color-1] -= penalty

        if board[n-2][n-1] == color:
            count = 0
            for i in range(2, n):
                if board[n-1-i][n-1] == color:
                    count += 1
                else:
                    break
            if count < n - 1:
                scores[color-1] -= penalty

    return scores

# consider mobility: immediate + potential
def update_mobility(board, color, scores, n_moves, disks):
    n = len(board)

    # immediate mobility, check the number of possible legal moves
    moves = get_possible_moves(board, color)

    # make the number of possible moves and the number of this player's disks weight more for opening and mid-game
    if n_moves < (n * n) - (n * n / 6):
        weight = 2
    else:
        weight = 1

    scores[color-1] += weight * len(disks)
    scores[color-1] += weight * len(moves)

    # potential mobility, current player want less disks that have empty disk adjacent
    for disk in disks:
        for xdir, ydir in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1],
                    [-1, 0], [-1, 1]]:
            row = disk[0] + xdir
            col = disk[1] + ydir
            if row >= 0 and row < n and col >= 0 and col < n and board[row][col] == 0:
                scores[color-1] -= 1
                break

    return scores

# Better heuristic value of board
# with board size 6*6, define opening be first 6 moves, end-game be last 6 moves, and
# mid-game be all middle moves (ignore off by 1 move)
def compute_heuristic(board, color):
    oppo = 3 - color
    n = len(board)

    # accumulate number of moves so far and the position of my disks and other disks
    n_moves = 0
    my_disks = []
    oppo_disks = []
    for i in range(n):
        for j in range(n):
            if board[i][j] == color:
                my_disks.append((i, j))
                n_moves += 1
            elif board[i][j] == oppo:
                oppo_disks.append((i, j))
                n_moves += 1

    # a list with first being the dark color's score, and second being the light color's score
    scores = [0, 0]

    # stable disk: corner and edge
    scores = stable_score(board, color, scores)
    scores = stable_score(board, oppo, scores)

    # X-square and isolated C-square
    scores = XC_square(board, color, scores, n_moves)
    scores = XC_square(board, oppo, scores, n_moves)

    # mobility
    scores = update_mobility(board, color, scores, n_moves, my_disks)
    scores = update_mobility(board, oppo, scores, n_moves, oppo_disks)

    # compute the difference and return as utility
    if color == 1:
        return scores[0] - scores[1]
    else:
        return scores[1] - scores[0]

############ MINIMAX ###############################
def minimax_min_node(board, color, limit, caching = 0):
    # Min node
    best_move = None
    oppo = 3 - color  # color will always be current player's color

    # if caching and the current board state has been visited
    if caching and board in caching_states:
        return best_move, caching_states[board]

    child_list = get_possible_moves(board, oppo)

    # if board is at terminal state or at depth limit
    if len(child_list) == 0 or limit == 0:
        return best_move, compute_utility(board, color)

    value = float("inf")

    for move in child_list:
        next_state = play_move(board, oppo, move[0], move[1])
        next_move, next_val = minimax_max_node(next_state, color, limit - 1, caching)

        # if caching, record the new board state and its minimax value to the dictionary caching_states
        if caching:
            caching_states[next_state] = next_val

        if value > next_val:
            value, best_move = next_val, move

    return best_move, value

def minimax_max_node(board, color, limit, caching = 0): #returns highest possible utility
    # MAX node
    best_move = None

    if caching and board in caching_states:
        return best_move, caching_states[board]

    child_list = get_possible_moves(board, color)

    # if board is at terminal state or at the depth limit
    if len(child_list) == 0 or limit == 0:
        return best_move, compute_utility(board, color)

    value = -float("inf")

    for move in child_list:
        next_state = play_move(board, color, move[0], move[1])
        next_move, next_val = minimax_min_node(next_state, color, limit - 1, caching)

        # record the new board state and its minimax value to the dictionary caching_states
        if caching:
            caching_states[next_state] = next_val

        if value < next_val:
            value, best_move = next_val, move

    return best_move, value

def select_move_minimax(board, color, limit, caching = 0):
    """
    Given a board and a player color, decide on a move.
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.
    """
    caching_states.clear()  # clear the global dictionary for future use

    # cast the list of lists to tuple of tuples for input board
    new_board = []
    for row in board:
        new_board.append(tuple(row))

    return minimax_max_node(tuple(new_board), color, limit, caching)[0]

############ ALPHA-BETA PRUNING #####################

# ordering nodex as follows:
# For MIN nodes, (child move yielding lowest value/utility) is explored first.
# For MAX nodes, (child move yielding highest value/utility) is explored first. 
def order_node(child_list, board, color, original_color):
    utility_move = {} # utility: [move]
    result = []

    # play_move for each successor, compute the utility and store move with same utility together in the dictionary
    for move in child_list:
        next_state = play_move(board, color, move[0], move[1])
        utility = compute_utility(next_state, original_color)
        if utility not in utility_move:
            utility_move[utility] = []
        utility_move[utility].append(move)

    # MAX node
    if color == original_color:
        for key in sorted(utility_move, reverse=True):
            result.extend(utility_move[key])

    # MIN node
    else:
        for key in sorted(utility_move):
            result.extend(utility_move[key])

    return result

def alphabeta_min_node(board, color, alpha, beta, limit, caching = 0, ordering = 0):
    best_move = None
    oppo = 3 - color

    if caching and board in caching_states:
        return best_move, caching_states[board]

    child_list = get_possible_moves(board, oppo)

    # if at terminal state or reach depth limit
    if len(child_list) == 0 or limit == 0:
        return best_move, compute_utility(board, color)

    value = float("inf")

    if ordering:
        child_list = order_node(child_list, board, oppo, color)

    for move in child_list:
        next_state = play_move(board, oppo, move[0], move[1])
        next_move, next_val = alphabeta_max_node(next_state, color, alpha, beta, limit - 1, caching, ordering)

        # record the new board state and its minimax value to the dictionary caching_states
        if caching:
            caching_states[next_state] = next_val

        if value > next_val:
            value, best_move = next_val, move
        if value <= alpha:
            break
        beta = min(beta, value)

    return best_move, value

def alphabeta_max_node(board, color, alpha, beta, limit, caching = 0, ordering = 0):
    best_move = None

    if caching and board in caching_states:
        return best_move, caching_states[board]

    child_list = get_possible_moves(board, color)

    # if at terminal state or reach depth limit
    if len(child_list) == 0 or limit == 0:
        return best_move, compute_utility(board, color)

    value = -float("inf")

    if ordering:
        child_list = order_node(child_list, board, color, color)

    for move in child_list:
        next_state = play_move(board, color, move[0], move[1])
        next_move, next_val = alphabeta_min_node(next_state, color, alpha, beta, limit - 1, caching, ordering)

        # record the new board state and its minimax value to the dictionary caching_states
        if caching:
            caching_states[next_state] = next_val

        if value < next_val:
            value, best_move = next_val, move
        if value >= beta:
            break
        alpha = max(alpha, value)

    return best_move, value

def select_move_alphabeta(board, color, limit, caching = 0, ordering = 0):
    """
    Given a board and a player color, decide on a move.
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.
    If ordering is ON (i.e. 1), use node ordering to expedite pruning and reduce the number of state evaluations.
    If ordering is OFF (i.e. 0), do NOT use node ordering to expedite pruning and reduce the number of state evaluations.
    """
    caching_states.clear()  # clear the global dictionary for future use

    # cast the list of lists to tuple of tuples for input board
    new_board = []
    for row in board:
        new_board.append(tuple(row))

    return alphabeta_max_node(tuple(new_board), color, -float("inf"), float("inf"), limit, caching, ordering)[0]

####################################################
def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("Othello AI") # First line is the name of this AI
    arguments = input().split(",")

    color = int(arguments[0]) #Player color: 1 for dark (goes first), 2 for light.
    limit = int(arguments[1]) #Depth limit
    minimax = int(arguments[2]) #Minimax or alpha beta
    caching = int(arguments[3]) #Caching
    ordering = int(arguments[4]) #Node-ordering (for alpha-beta only)

    if (minimax == 1): eprint("Running MINIMAX")
    else: eprint("Running ALPHA-BETA")

    if (caching == 1): eprint("State Caching is ON")
    else: eprint("State Caching is OFF")

    if (ordering == 1): eprint("Node Ordering is ON")
    else: eprint("Node Ordering is OFF")

    if (limit == -1): eprint("Depth Limit is OFF")
    else: eprint("Depth Limit is ", limit)

    if (minimax == 1 and ordering == 1): eprint("Node Ordering should have no impact on Minimax")

    while True: # This is the main loop
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        # The first number is the score for player 1 (dark), the second for player 2 (light)
        next_input = input()
        status, dark_score_s, light_score_s = next_input.strip().split()
        dark_score = int(dark_score_s)
        light_score = int(light_score_s)

        if status == "FINAL": # Game is over.
            print
        else:
            board = eval(input()) # Read in the input and turn it into a Python
                                  # object. The format is a list of rows. The
                                  # squares in each row are represented by
                                  # 0 : empty square
                                  # 1 : dark disk (player 1)
                                  # 2 : light disk (player 2)

            # Select the move and send it to the manager
            if (minimax == 1): #run this if the minimax flag is given
                movei, movej = select_move_minimax(board, color, limit, caching)
            else: #else run alphabeta
                movei, movej = select_move_alphabeta(board, color, limit, caching, ordering)

            print("{} {}".format(movei, movej))

if __name__ == "__main__":
    run_ai()
