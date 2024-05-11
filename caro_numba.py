"""
    Import Thư viện
"""
from tqdm import tqdm
import time
from numba import types, typed, int64, optional, deferred_type, float64
from numba.experimental import jitclass
import typing as pt
import numpy as np
import random as rd
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# Notice that NUMBER_ROWS = NUMBER_COLS

torch.manual_seed(0)
from numba import njit
import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaExperimentalFeatureWarning, NumbaWarning

warnings.simplefilter('ignore', category = NumbaDeprecationWarning)
warnings.simplefilter('ignore', category = NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category = NumbaExperimentalFeatureWarning)
warnings.simplefilter('ignore', category = NumbaWarning)

"""
    Môi trường của game
"""

# Các hằng số tùy biến của trò chơi
NUMBER_ROWS = 15
NUMBER_COLS = 15
NUMBER_PLAYERS = 2
NUMBER_ACTIONS = NUMBER_ROWS * NUMBER_COLS
ENV_SIZE = NUMBER_ROWS * NUMBER_COLS + 3
STATE_SIZE = NUMBER_ROWS * NUMBER_COLS + 3

# Hàm khởi tạo môi trường bàn cờ
@njit()
def init_env():
    env_state = np.full(ENV_SIZE, 0)
    return env_state

# Hàm trả về kích cỡ môi trường bàn cờ
@njit()
def get_state_size():
    return STATE_SIZE

# Hàm trả về số nước đi có thể của trò chơi
@njit()
def get_action_size():
    return NUMBER_ACTIONS

# Hàm trả về số lượng người chơi
@njit()
def get_agent_size():
    return NUMBER_PLAYERS

# Hàm đổi nước đi tọa độ 2D của bàn cờ [15][15] sang nước đi 1D của bàn cờ [255]
@njit()
def convert_to_1D(i, j):
    return i * NUMBER_COLS + j

# Hàm đổi nước đi tọa độ 1D của bàn cờ [255] sang nước đi 2D của bàn cờ [15][15]
@njit()
def convert_to_2D(act):
    x = int(act / NUMBER_COLS)
    y = act - x * NUMBER_COLS
    return x, y

# Hàm trả về người tiếp theo tới lượt
@njit()
def get_opponent_player(player):
    return 1 if player == 0 else 0

# Hàm trả về giá trị bàn cờ đối với đối phương
@njit()
def get_opponent_value(value):
    return -value

# Hàm trả về state mà mỗi agent có thể lấy
@njit()
def get_agent_state( env_state):
    p_state = np.full(STATE_SIZE, 0)
    # Get board state
    p_state[0 : (NUMBER_ROWS * NUMBER_COLS)] = env_state[0 : (NUMBER_ROWS * NUMBER_COLS)]
    # Get last checked cell
    p_state[NUMBER_ROWS * NUMBER_COLS] = env_state[NUMBER_ROWS * NUMBER_COLS]
    p_state[NUMBER_ROWS * NUMBER_COLS + 1] = env_state[NUMBER_ROWS * NUMBER_COLS + 1]
    return p_state

# Hàm trả về các nước đi hợp lệ ( = 1 if valid else 0 )
@njit()
def get_valid_actions(player_state):
    # list_action = np.full(NUMBER_ACTIONS, 0)
    # list_action[np.where(player_state[0 : NUMBER_ROWS * NUMBER_COLS] == 0)] = 1
    # return list_action
    return (player_state[0 : NUMBER_ACTIONS] == 0).astype(np.uint8)

# Hàm kiểm tra hết cờ hay chưa
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

# Hàm trả về môi trường sau khi đánh nước đi action
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

"""
    Bot lấy các nước đi random
"""
@njit()
def numba_bot_random(p_state, per):
    arr_action = get_valid_actions(p_state)
    act_idx = np.random.choice(np.where(arr_action == 1)[0])
    return act_idx, per

"""
    Hàm khởi chạy một game cờ caro, với 2 agent là p_main và p_o
"""
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

"""
    Hàm chạy num_game game
"""
@njit()
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

# Hàm encode môi trường về channel [3][15][15] sử dụng trong mạng nn
@njit()
def get_encode_state(env):
    env = np.reshape(env[ : 225], (15, 15))
    encode_state = np.stack(
        (env == 2, env == 0, env == 1)
    ).astype(np.float32)
    return encode_state

# Hàm đổi góc nhìn về người chơi đánh nước tới luôn luôn là x
@njit()
def change_perspective(env, player):
    n_env = np.copy(env)
    if player == 0: return n_env
    temp = np.where(n_env[0 : NUMBER_ROWS * NUMBER_COLS] == 2)
    n_env[np.where(n_env[0 : NUMBER_ROWS * NUMBER_COLS] == 1)] = 2
    n_env[temp] = 1
    return n_env

kv_ty = (types.unicode_type, types.int64)
kv_ty1 = (types.unicode_type, types.float64)
node_type = deferred_type()
node_type1 = deferred_type()


"""
    Các class Node là các node trong cây tìm kiếm MCTS, class Child nhằm khắc phục việc không thể sử dụng jitclass trong chính class đó
"""
@jitclass
class Node:
    args : types.DictType(*kv_ty) # type: ignore
    args_f : types.DictType(*kv_ty1) # type: ignore
    env : int64[:] # type: ignore
    parent : optional(node_type) # type: ignore
    action_taken : int64 # type: ignore
    prior : float64 # type: ignore
    children : optional(node_type1) # type: ignore
    visit_count : int64 # type: ignore
    value_sum : float64 # type: ignore
    def __init__(self, args, args_f, env, parent = None, action_taken = -1, prior = 0.0, visit_count = 0):
        self.args = args
        self.args_f = args_f
        self.env = env
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.children = None
        self.visit_count = visit_count
        self.value_sum = 0.0

    def is_fully_expanded(self):
        return self.children is not None

    def select(self):
        return self.children.select_child(self.visit_count)


@jitclass
class Childs:
    childs: types.ListType(Node.class_type.instance_type) # type: ignore
    def __init__(self, node):
        self.childs = typed.List([node])

    def append_node(self, node):
        self.childs.append(node)

    def select_child(self, pr_vc):
        best_child = None
        best_ucb = -np.inf

        for child in self.childs:
            ucb = self.cal_ucb(child, pr_vc)
            if best_ucb < ucb:
                best_child = child
                best_ucb = ucb
        return best_child

    def print_node(self):
        for i, child in enumerate(self.childs):
            print(i, child.prior)
            print()

    def cal_ucb(self, node, pr_vc):
        if(node.visit_count == 0):
            q_value = 0
        else:
            q_value = 1 - ((node.value_sum / node.visit_count) + 1) / 2
        return q_value + node.args['C'] * (math.sqrt(pr_vc) / (node.visit_count + 1)) * node.prior

    def get_action_probs(self):
        action_probs = np.zeros(get_action_size())
        for child in self.childs:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs

node_type.define(Node.class_type.instance_type)
node_type1.define(Childs.class_type.instance_type)

def expand(root, policy):
    for act, prob in enumerate(policy):
        if prob > 0:
            child_env = next_step(act, root.env)
            child_env = change_perspective(child_env, 1)
            child = Node(root.args, root.args_f, child_env, root, act, prob)
            if root.children is not None:
                root.children.append_node(child)
            else:
                root.children = Childs(child)

def backpropagate(node, value):
    node.visit_count += 1
    node.value_sum += value
    if node.parent is not None:
        backpropagate(node.parent, -value)


class MCTS:
    def __init__(self, args, args_f, model) -> None:
        # args: chứa những thông tin để sử dụng hoặc kết thúc thuật toán
        self.args = args
        self.args_f = args_f
        self.model = model

    # Không dùng data để train ngay lập tức mà chỉ sử dụng resnet để dự đoán policy và value
    @torch.no_grad()
    def search(self, env):
        root = Node(self.args, self.args_f, env, visit_count = 1)


        # Thêm noise, tăng khả năng tìm kiếm
        policy, _ = self.model(
            torch.tensor(get_encode_state(env), device = self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis = 1).squeeze(0).cpu().numpy()

        policy = (1 - self.args_f['dirichlet_esp']) * policy + self.args_f['dirichlet_esp'] * np.random.dirichlet([self.args_f['dirichlet_alp']] * NUMBER_ACTIONS)


        valid_acts = get_valid_actions(env)
        policy *= valid_acts
        policy /= np.sum(policy)

        expand(root, policy)
        for _search in range (self.args['num_searches']):
            # Selection phase:
            node = root
            while node.is_fully_expanded():
                node = node.select()

            # Prepare for next step
            check_win = check_ended(node.env) if not node.action_taken == -1 else -1
            ## Value ở đây luôn mang giá trị -1, vì người chơi hiện tại chưa đánh mà bàn cờ đã kết thúc => thua
            value = 0
            if check_win == 0 or check_win == 1:
                value = -1
            if check_win == -1:
                # Dùng mạng đưa ra policy và value
                policy, value = self.model(
                    torch.tensor(get_encode_state(node.env), device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, 1).squeeze(0).cpu().numpy()
                policy = convert_policy(policy, node.env)
                value = value.item()
                # Expansion phase:
                expand(node, policy)
            # Backpropagation phase
            backpropagate(node, value)

        # Return the prior probalities
        action_probs = root.children.get_action_probs()
        return action_probs

@njit()
def convert_policy(policy, env):
    valid_move = get_valid_actions(env)
    policy *= valid_move
    policy /= np.sum(policy)
    return policy

class ResNet(nn.Module):
    def __init__(self, num_resBlock, num_hidden, device) -> None:
        super().__init__()
        self.device = device
        # Layer Input
        ## Chuyển hóa encode env thành các input cho layer backBone
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )

        # Layer backBone bao gồm các RestBlock
        self.backBone = nn.ModuleList(
            [RestBlock(num_hidden) for _n in range(num_resBlock)]
        )

        ## Huấn luyện động thời 2 layer
        # Layer output policyHead đưa ra prior probability
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * NUMBER_ROWS * NUMBER_COLS, NUMBER_ACTIONS)
        )

        # Layer output đưa ra giá trị của env
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * NUMBER_ROWS * NUMBER_COLS, 1),
            nn.Tanh()
        )

        self.to(device)

    # Hàm mô quá đường đi của dữ encode env x khi xử lý qua mạng
    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock.forward(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value

# Các block liên tiếp trong layer backBone của cấu trúc Residual neural network
class RestBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.batchn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.batchn2 = nn.BatchNorm2d(num_hidden)

    # _x được lấy dư và cộng đồng thời với output thông qua xử lý của RestBlock
    def forward(self, x):
        residual = x
        x = F.relu(self.batchn1(self.conv1(x)))
        x = self.batchn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

# Hàm hiển thị bàn cờ
def print_env(env):
    for i in range(NUMBER_COLS):
        print(env[i * 15 : i * 15 + 15])
    print(env[NUMBER_ROWS * NUMBER_COLS : NUMBER_ROWS * NUMBER_COLS + 3])

"""
    Class agent bot AlphaGomoku
    Bao gồm các hàm tự học, train
"""
class AlphaGomoku:
    def __init__(self, model, optimizer, args, args_f):
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.args_f = args_f
        self.mcts = MCTS(args, args_f, model)

    def selfPlay(self):
        memory = []
        env = init_env()
        player = 0
        while True:
              neutral_env = change_perspective(env, player)
              action_prob = self.mcts.search(neutral_env)
              memory.append((neutral_env, action_prob, player))

              # temp_probs nhằm tăng hoặc giảm khoảng cách xác suất giữa các nước đi
              ## Điều chỉnh để có thể ưu tiên tìm kiếm hoặc ưu tiên khai thác
              temperature_action_probs = get_temp_act_probs(action_prob, self.args_f['temperature'])
              temperature_action_probs /= np.sum(temperature_action_probs)
              action = np.random.choice(NUMBER_ACTIONS, p=temperature_action_probs)
              env = next_step(action, env)
              value = check_ended(env)

              if value != -1:
                  value = 0 if value == 2 else 1
                  returnMemory = []
                  for his_env, his_act_prob, his_player in memory:
                      his_outcome = value if his_player == player else get_opponent_value(value)
                      returnMemory.append((
                          get_encode_state(his_env),
                          his_act_prob,
                          his_outcome
                      ))
                  return returnMemory
              player = get_opponent_player(player)

    def train(self, memory):
        random.shuffle(memory)
        for batch_ind in range(0, len(memory), self.args['batch_size']):
            sample = memory[batch_ind : min(len(memory) - 1, batch_ind + self.args['batch_size'])]
            states, policy_targets, value_targets = zip(*sample)

            states, policy_targets, value_targets = np.array(states), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)

            states = torch.tensor(states, dtype = torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype = torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype = torch.float32, device=self.model.device)

            out_policy, out_value = self.model(states)


            # Đánh giá sự mất mát của hàm cross_entropy ( hàm đo lường mức độ tương tự giữa phân phối xác suất của mạng đưa ra và phân phối thực tế )
            policy_loss = F.cross_entropy(out_policy, policy_targets)

            # Đánh giá sự mất mát theo phương thức bình phương sai số trung bình của giá trị bàn cờ giữa giá trị thực và giá trị của mạng đưa ra
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            # Điều chỉnh gradient về 0
            self.optimizer.zero_grad()
            # Tính toán gradient và lan truyền ngược
            loss.backward()
            # Cập nhật trọng số của mô hình
            self.optimizer.step()


    def learn(self):
        for i in range(self.args['num_iterations']):
            memory = []
            self.model.eval()
            for selfPlay_i in tqdm(range(self.args['num_selfPlay_iterations'])):
                memory += self.selfPlay()

            self.model.train()
            for epoch in tqdm(range(self.args['num_epochs'])):
                self.train(memory)

            torch.save(self.model.state_dict(), f"model_{i}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{i}.pt")

    ### Bổ sung thêm hàm chơi với các agent khác!
    def play_with_another_agent(self, agent):
        pass

# Hàm làm mượn hoặc tăng khoảng cách các phân phối xác suất của các nước đi dựa trên temp
@njit()
def get_temp_act_probs(action_prob, t):
    return action_prob ** (1 / t)


"""
    Hàm trả về agent AlphaGomoku
"""

def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Có thể sử dụng load_state_dict để lấy model cũ tiếp tục huấn luyện
    # model.load_state_dict(torch.load('model_0.pt'))
    model = ResNet(4, 64, device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.0001)

    # Điều chỉnh các tham số phù hợp
    args = typed.Dict()
    args['C'] = 2
    args['num_searches'] = 20
    args['num_iterations'] = 2
    args['num_selfPlay_iterations'] = 5
    args['num_epochs'] = 4
    args['batch_size'] = 64

    args_f = typed.Dict()
    args_f['temperature'] = 1.25
    args_f['dirichlet_esp'] = 0.25
    args_f['dirichlet_alp'] = 0.35


    alphaGomoku = AlphaGomoku(model, optimizer, args, args_f)
    return alphaGomoku
    # alphaGomoku.learn()

## Test MCTS
args = typed.Dict()
args['C'] = 2
args['num_searches'] = 20
args['num_iterations'] = 2
args['num_selfPlay_iterations'] = 5
args['num_epochs'] = 4
args['batch_size'] = 64

args_f = typed.Dict()
args_f['temperature'] = 1.25
args_f['dirichlet_esp'] = 0.25
args_f['dirichlet_alp'] = 0.35
def one_game_pvc():
    model = ResNet(4, 64, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    mcts = MCTS(args, args_f, model)
    env = init_env()
    while True:
        if env[NUMBER_ROWS * NUMBER_COLS + 2] % 2 == 0:
            valid_moves = get_valid_actions(env)
            act = int(input("Choose action: "))
            if valid_moves[act] == 0:
                print("action not valid")
                continue
        else:
            neutral_state = change_perspective(env, 1)
            mcts_probs = mcts.search(neutral_state)
            act = np.argmax(mcts_probs)

        env = next_step(act, env)
        check_end = check_ended(env)
        if check_end != -1:
            if check_end == 2:
                print('\n---------------------- All tie! ----------------------')
            elif check_end == 0:
                print('\n---------------------- Winner: Human ----------------------')
            elif check_end == 1:
                print('\n---------------------- Winner: Comp ----------------------')

            break
one_game_pvc()