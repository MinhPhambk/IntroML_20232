import numpy as np
import random as rd

# Notice that NUMBER_ROWS = NUMBER_COLS
NUMBER_ROWS = 15
NUMBER_COLS = 15

NUMBER_PLAYERS = 2
NUMBER_ACTIONS = NUMBER_ROWS * NUMBER_COLS
ENV_SIZE = NUMBER_ROWS * NUMBER_COLS + 3
STATE_SIZE = NUMBER_ROWS * NUMBER_COLS + 3

from numba import njit
import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaExperimentalFeatureWarning, NumbaWarning

warnings.simplefilter('ignore', category = NumbaDeprecationWarning)
warnings.simplefilter('ignore', category = NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category = NumbaExperimentalFeatureWarning)
warnings.simplefilter('ignore', category = NumbaWarning)

@njit()
def init_env():
    env_state = np.full(ENV_SIZE, 0)
    return env_state

@njit()
def get_state_size():
    return STATE_SIZE

@njit()
def get_action_size():
    return NUMBER_ACTIONS

@njit()
def get_agent_size():
    return NUMBER_PLAYERS

@njit()
def convert_to_1D(i = NUMBER_ROWS - 1, j = NUMBER_COLS - 1):
    return i * NUMBER_COLS + j

@njit()
def convert_to_2D(act):
    x = int(act / NUMBER_COLS)
    y = act - x * NUMBER_COLS
    return x, y

@njit()
def get_agent_state(env_state):
    p_state = np.full(STATE_SIZE, 0)

    # Get board state
    p_state[0 : (NUMBER_ROWS * NUMBER_COLS)] = env_state[0 : (NUMBER_ROWS * NUMBER_COLS)]

    # Get last checked cell
    p_state[NUMBER_ROWS * NUMBER_COLS] = env_state[NUMBER_ROWS * NUMBER_COLS]
    p_state[NUMBER_ROWS * NUMBER_COLS + 1] = env_state[NUMBER_ROWS * NUMBER_COLS + 1]

    return p_state

@njit()
def get_valid_actions(player_state):
    list_action = np.full(NUMBER_ACTIONS, 0)

    for i in range(NUMBER_ROWS):
        for j in range(NUMBER_COLS):
            id = convert_to_1D(i, j)
            if (player_state[id] == 0):
                list_action[id] = 1

    return list_action

@njit()
def check_ended(env):
    turn = env[NUMBER_ROWS * NUMBER_COLS + 2] - 1
    
    # Case 1: all tie
    check_sum = 0
    for i in range(NUMBER_COLS):
        for j in range(NUMBER_ROWS):
            if (env[convert_to_1D(i, j)] != 0): check_sum += 1
                
    if (check_sum == NUMBER_ROWS * NUMBER_COLS):
        return 2

    # Case 2:
    else:
        p_id = turn % 2
        x = env[NUMBER_ROWS * NUMBER_COLS]
        y = env[NUMBER_ROWS * NUMBER_COLS + 1]

        # Check row
        count = 1
        d = 1
        while y + d < NUMBER_COLS and env[convert_to_1D(x, y + d)] == p_id + 1:
            count += 1
            if count == 5:
                return p_id
            d += 1
        d = 1
        while y - d > -1 and env[convert_to_1D(x, y - d)] == p_id + 1:
            count += 1
            if count == 5:
                return p_id
            d += 1

        # Check col
        count = 1
        d = 1
        while x + d < NUMBER_ROWS and env[convert_to_1D(x + d, y)] == p_id + 1:
            count += 1
            if count == 5:
                return p_id
            d += 1
        d = 1
        while x - d > -1 and env[convert_to_1D(x - d, y)] == p_id + 1:
            count += 1
            if count == 5:
                return p_id
            d += 1

        # Check diagonal line C1
        count = 1
        d = 1
        while x + d < NUMBER_ROWS and y + d < NUMBER_COLS and env[convert_to_1D(x + d, y + d)] == p_id + 1:
            count += 1
            if count == 5:
                return p_id
            d += 1
        d = 1
        while x - d > -1 and y - d > -1 and env[convert_to_1D(x - d, y - d)] == p_id + 1:
            count += 1
            if count == 5:
                return p_id
            d += 1

        # Check diagonal line C2
        count = 1
        d = 1
        while x + d < NUMBER_ROWS and y - d > -1 and env[convert_to_1D(x + d, y - d)] == p_id + 1:
            count += 1
            if count == 5:
                return p_id
            d += 1
        d = 1
        while x - d > -1 and y + d < NUMBER_COLS and env[convert_to_1D(x - d, y + d)] == p_id + 1:
            count += 1
            if count == 5:
                return p_id
            d += 1

    return -1

@njit()
def next_step(action, env_state):
    env = np.copy(env_state)
    actions = get_valid_actions(get_agent_state(env))

    if actions[action] == 0:
        raise Exception('Action error!')
    else: 
        p_id = env[NUMBER_ROWS * NUMBER_COLS + 2] % 2
        env[action] = p_id + 1
        env[NUMBER_ROWS * NUMBER_COLS + 2] += 1

    x, y = convert_to_2D(action)
    env[NUMBER_ROWS * NUMBER_COLS] = x
    env[NUMBER_ROWS * NUMBER_COLS + 1] = y

    return env

@njit
def numba_bot_random(p_state, per):
    arr_action = get_valid_actions(p_state)
    arr_action = np.where(arr_action == 1)[0]
    act_idx = np.random.randint(0, len(arr_action))

    return arr_action[act_idx], per

@njit()
def numba_run_one_game(p_main, p_o, per, print_mode = False):
    env = init_env()

    _cc = 0
    while _cc <= NUMBER_COLS * NUMBER_ROWS:            
        p_idx = env[NUMBER_COLS * NUMBER_ROWS + 2] % 2
        p_state = get_agent_state(env)
        turn = env[NUMBER_COLS * NUMBER_ROWS + 2]

        if (print_mode):
            print('----------------------------------------------------------------------------------')
            if (turn % 2 == 0):
                print('Turn of player: X')
            elif (turn % 2 == 1):
                print('Turn of player: O')
        
        if (p_idx == 0):
            action, per = p_main(p_state, per)
        elif (p_idx == 1):
            action, per = p_o(p_state, per)

        env = next_step(action, env)
        
        if (print_mode):
            print('Checked cell: (', env[NUMBER_ROWS * NUMBER_COLS], ',', env[NUMBER_ROWS * NUMBER_COLS + 1], ')')
        
        _cc += 1        
        
        if (check_ended(env) != -1):
            break


    if (print_mode):
        if check_ended(env) == 2:
            print('\n---------------------- All tie! ----------------------')
        elif check_ended(env) == 0:
            print('\n---------------------- Winner: X ----------------------')
        elif check_ended(env) == 1:
            print('\n---------------------- Winner: O ----------------------')

    winner = 0
    if (check_ended(env) == 2):
        winner = -1
    elif (check_ended(env) == 1):
        winner = 1
    else:
        winner = 0

    return winner, per

@njit
def numba_run_n_game(p0, p1, per, num_game, print_mode = False):
    win = [0, 0]
    for _n in range(num_game):
        first = rd.randint(0, 1)
        if (first == 0):
            winner, per = numba_run_one_game(p0, p1, per, print_mode)
        else:
            winner, per = numba_run_one_game(p1, p0, per, print_mode)
        
        if winner != -1:
            if (winner == 0):
                win[0] += 1 * (1 - first)
                win[1] += 1 * first
            elif (winner == 1):
                win[0] += 1 * first
                win[1] += 1 * (1 - first)
            
    if (print_mode):
        print()
        
    return win, per