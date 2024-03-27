
import numpy as np
import random as rd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# Notice that NUMBER_ROWS = NUMBER_COLS

class ResNet(nn.Module):
    def __init__(self, num_resBlock) -> None:
        pass
        

class Node:
    def __init__(self, args, env, parent = None, action_taken = None) -> None:
        # args: chứa những thông tin để sử dụng hoặc kết thúc thuật toán
        self.args = args
        # env: môi trường ( state cũng được ) của node
        self.env = env
        # parent: cha của node hiện tại
        self.parent = parent
        # action_taken: từ node cha đánh nước đi act này được node con
        self.action_taken = action_taken

        # children: list các node con của node hiện tại
        self.children = []
        # expandable_moves: các nước đi có thể đi tính từ node hiện tại
        self.expandable_moves = get_valid_actions(self.env)

        ## Variable MCTS algorithm
        # visit_count: số lần đã đi qua node hiện tại
        self.visit_count = 0
        # value_sum: giá trị của bàn cờ tính bởi thuật toán 
        self.value_sum = 0

    # Hàm kiểm tra xem node hiện tại có phải là node mở rộng hoàn toàn rồi hay chưa
    def is_fully_expanded(self):
        return np.sum(self.expandable_moves) == 0 and len(self.children) > 0

    # Thuật toán chọn node có UCB lớn nhất
    def select(self):
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.cal_ucb(child)
            if best_ucb < ucb:
                best_child = child
                best_ucb = ucb

        return best_child
    
    # Hàm tính giá trị UCB của node
    def cal_ucb(self, child):
        # Cần đưa đối thủ vào trạng thái bất lợi nhất, tức q_value lớn nhất.
        ## + 1 rồi / 2 để giữ cho q_value trong [0, 1]
        q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)

    # Hàm mở rộng hay thêm một nốt con vào nốt hiện tại
    def expand(self):
        act = np.random.choice(np.where(self.expandable_moves == 1)[0])
        self.expandable_moves[act] = 0
        child_env = next_step(act, self.env)
        child_env = change_perspective(child_env)
        child = Node(self.args, child_env, self, act)
        self.children.append(child)
        return child

    # Hàm mô phỏng bàn cờ với các nước đi ngẫu nhiên cho tới cuối cùng
    def simulation(self):
        check_win = check_ended(self.env) if not self.action_taken == None else -1
        if check_win == 1 or check_win == 0:
            ## Return value ở đây luôn mang giá trị -1, vì người chơi hiện tại chưa đánh mà bàn cờ đã kết thúc => thua
            return -1
            # return -1
        rollout_env = np.copy(self.env)
        valid_move = get_valid_actions(rollout_env)
        while True:
            act = np.random.choice(np.where(valid_move == 1)[0])
            valid_move[act] = 0
            rollout_env = next_step(act, rollout_env)
            check_win = check_ended(rollout_env)
            ## Return lại value đúng giá trị kết quả của bàn cờ.
            if check_win != -1:
                if check_win == 1:
                    return -1
                elif check_win == 2:
                    return 0
                else:
                    return 1
    # Hàm truyền ngược các giá trị value và visit_count lên root node
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        if self.parent is not None:
            self.parent.backpropagate(-value)

class MCTS:
    def __init__(self, args) -> None:
        # args: chứa những thông tin để sử dụng hoặc kết thúc thuật toán
        self.args = args


    def search(self, env):
        root = Node(self.args, env)
        for _search in range (self.args['num_searches']):
            # Selection phase:
            node = root
            while node.is_fully_expanded():
                node = node.select()

            # Prepare for next step
            check_win = check_ended(node.env) if not node.action_taken == None else -1
            ## Value ở đây luôn mang giá trị -1, vì người chơi hiện tại chưa đánh mà bàn cờ đã kết thúc => thua
            value = 0
            if check_win == 0 or check_win == 1:
                value = -1


            if check_win == -1:
                # Expansion phase:
                node = node.expand()
                # Simulation phase
                value = node.simulation()
            # Backpropagation phase
            node.backpropagate(value)
        
        # Return the prior probalities
        action_probs = np.zeros(get_action_size())
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
                
        

from numba import njit
import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaExperimentalFeatureWarning, NumbaWarning

warnings.simplefilter('ignore', category = NumbaDeprecationWarning)
warnings.simplefilter('ignore', category = NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category = NumbaExperimentalFeatureWarning)
warnings.simplefilter('ignore', category = NumbaWarning)

args = {
    'C' : 1.41,
    'num_searches': 1000
}

NUMBER_ROWS = 15
NUMBER_COLS = 15
NUMBER_PLAYERS = 2
NUMBER_ACTIONS = NUMBER_ROWS * NUMBER_COLS
ENV_SIZE = NUMBER_ROWS * NUMBER_COLS + 3
STATE_SIZE = NUMBER_ROWS * NUMBER_COLS + 3

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
def convert_to_1D(i, j):
    return i * NUMBER_COLS + j

@njit()
def convert_to_2D(act):
    x = int(act / NUMBER_COLS)
    y = act - x * NUMBER_COLS
    return x, y

@njit()
def get_agent_state( env_state):
    p_state = np.full(STATE_SIZE, 0)
    # Get board state
    p_state[0 : (NUMBER_ROWS * NUMBER_COLS)] = env_state[0 : (NUMBER_ROWS * NUMBER_COLS)]
    # Get last checked cell
    p_state[NUMBER_ROWS * NUMBER_COLS] = env_state[NUMBER_ROWS * NUMBER_COLS]
    p_state[NUMBER_ROWS * NUMBER_COLS + 1] = env_state[NUMBER_ROWS * NUMBER_COLS + 1]
    return p_state

@njit()
def get_valid_actions(player_state):
    # list_action = np.full(NUMBER_ACTIONS, 0)
    # list_action[np.where(player_state[0 : NUMBER_ROWS * NUMBER_COLS] == 0)] = 1
    # return list_action
    return (player_state[0 : NUMBER_ACTIONS] == 0).astype(np.uint8)

@njit()
def check_ended(env):
    
    # Case 1: check end
    # p_id = turn % 2
    x = env[NUMBER_ROWS * NUMBER_COLS]
    y = env[NUMBER_ROWS * NUMBER_COLS + 1]
    p_id = env[x * NUMBER_COLS + y] - 1
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
    # Case 2: all tie
    if(env[NUMBER_ROWS * NUMBER_COLS + 2] == NUMBER_ROWS * NUMBER_COLS):
        return 2
    return -1

@njit()
def next_step(action, env_state):
    env = np.copy(env_state)
    if env[action] != 0:
        raise Exception('Action error!')
    else: 
        x = env[NUMBER_ROWS * NUMBER_COLS]
        y = env[NUMBER_ROWS * NUMBER_COLS + 1]
        p_id = env[x * NUMBER_COLS + y] % 2
        env[action] = p_id + 1
        env[NUMBER_ROWS * NUMBER_COLS + 2] += 1

    x, y = convert_to_2D(action)
    env[NUMBER_ROWS * NUMBER_COLS] = x
    env[NUMBER_ROWS * NUMBER_COLS + 1] = y

    return env

@njit
def numba_bot_random(p_state, per):
    arr_action = get_valid_actions(p_state)
    act_idx = np.random.choice(np.where(arr_action == 1)[0])
    return act_idx, per

@njit()
def numba_run_one_game(p_main, p_o, per, print_mode = False):
    env = init_env()
    _cc = 0
    while _cc < NUMBER_COLS * NUMBER_ROWS:            
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
        
    winner = check_ended(env)
    if (print_mode):
        if winner == 2:
            print('\n---------------------- All tie! ----------------------')
        elif winner == 0:
            print('\n---------------------- Winner: X ----------------------')
        elif winner == 1:
            print('\n---------------------- Winner: O ----------------------')
    
    if (winner == 2):
        winner = -1
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

## Hàm này biến đổi góc nhìn env (hay state) của bàn cờ đối với mỗi node
## Với mỗi Node, ta coi như lượt đi hiện tại là x (hay 1)?? 
@njit
def change_perspective(env):
    n_env = np.copy(env)
    temp = np.where(n_env[0 : NUMBER_ROWS * NUMBER_COLS] == 2)
    n_env[np.where(n_env[0 : NUMBER_ROWS * NUMBER_COLS] == 1)] = 2
    n_env[temp] = 1
    return n_env

def print_env(env):
    for i in range(NUMBER_COLS):
        print(env[i * 15 : i * 15 + 15])
    print(env[NUMBER_ROWS * NUMBER_COLS : NUMBER_ROWS * NUMBER_COLS + 3])

def one_game_pvc():
    mcts = MCTS(args)
    env = init_env()
    while True:
        print_env(env)
        if env[NUMBER_ROWS * NUMBER_COLS + 2] % 2 == 0:
            valid_moves = get_valid_actions(env)
            print(valid_moves)
            act = int(input("Choose action: "))
            if valid_moves[act] == 0:
                print("action not valid")
                continue
        else:
            neutral_state = change_perspective(env)
            mcts_probs = mcts.search(neutral_state)
            act = np.argmax(mcts_probs)

        env = next_step(act, env)
        check_end = check_ended(env)
        if check_end != -1:
            print_env(env)
            if check_end == 2:
                print('\n---------------------- All tie! ----------------------')
            elif check_end == 0:
                print('\n---------------------- Winner: Human ----------------------')
            elif check_end == 1:
                print('\n---------------------- Winner: Comp ----------------------')
            
            break
        
# one_game_pvc()
