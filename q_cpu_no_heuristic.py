import numpy as np
import time
from caro_cpu import get_valid_actions, next_step, check_ended, NUMBER_COLS, NUMBER_ROWS, next_step

#khởi tạo điểm cho các trường hợp thắng, thua, hoà
WIN_REWARD = 100
LOSE_REWARD = -100
TIE_REWARD = -10
NOT_END = 0
LEARNING_RATE = 0.8
DISCOUNT_FACTOR = 0.9
EPSILON = 0.3
DECAY_RATE = 0.95
RANGE = NUMBER_COLS * NUMBER_ROWS
LAMDA = 10

q_table = {} # rỗng
def make_random_q_table(n):
    q_table["0"]= {0:LAMDA}
    for i in range(n):
        #tạo mảng 228 phần tử ngẫu nhiên từ 0-2
        state = np.random.randint(0, 3, NUMBER_COLS * NUMBER_ROWS + 3)
        #chuyển RANGE kí tự đầu tiên thành string để làm key
        q_idx = str(state[:RANGE]).replace(" ", "").replace("[", "").replace("]", "").replace("\n", "")
        #mỗi key q_idx bao gồm các cặp key-value, key là action, value là điểm số
        q_table[q_idx] = make_new_q_value(state)
    return q_table

#tạo 1 dòng Q cho trạng thái mới, nếu trạng thái là trạng thái kết thúc thì trả về điểm số cho trạng thái đó
def make_new_q_value(p_state):
    new_q_value = {}
    p_id = p_state[NUMBER_ROWS*NUMBER_COLS+2]%2
    new_q_value_score = 0
    
    if check_ended(p_state) == 2:
        new_q_value_score = TIE_REWARD
    else:
        if check_ended(p_state) == -1:
            new_q_value_score = NOT_END
        else:
            if check_ended(p_state) == p_id:
                new_q_value_score = WIN_REWARD
            else:
                new_q_value_score = LOSE_REWARD

    #tạo ra dòng mới với điểm số cho từng action = điểm số của trạng thái mới * 1000000
    if new_q_value_score != NOT_END:
        new_q_value[1] = new_q_value_score
        return new_q_value
    for i in range(NUMBER_COLS * NUMBER_ROWS):
        if p_state[i] == 0:
            new_q_value[i] = new_q_value_score
        #NẾU dòng mới rỗng thì tạo ra 1 dòng mới với 1 action = 0
    if len(new_q_value) == 0:
        new_q_value[1] = 0
    return new_q_value

#bot q-learning không sử dụng heuristic, chỉ tính điểm khi đạt trạng thái đích
def q_bot_cpu_noH(p_state, per):

    arr_action = get_valid_actions(p_state)
    arr_action = np.where(arr_action == 1)[0]
    p_id = p_state[NUMBER_ROWS*NUMBER_COLS+2]%2
    # giảm hệ só khai phá trên mỗi nước đi bằng hàm mũ
    epsilon = EPSILON* (DECAY_RATE ** (NUMBER_COLS*NUMBER_ROWS -len(arr_action)))
    #nếu nước đi của O thì cần chuyển đổi đồng nhất về X
    if(p_id):
    #chuyển đổi từ nước đi 0 thành X: các nước đi 1 thành 2 và ngược lại
        xx= p_state[RANGE];
        yy= p_state[RANGE+1];
        turn= p_state[RANGE+2];
        p_state[p_state == 1]=3;
        p_state[p_state == 2]=1;
        p_state[p_state == 3]=2;
        p_state[RANGE]=xx;
        p_state[RANGE+1]=yy;
        p_state[RANGE+2]= 1 - turn;
    
    q_idx = str(p_state[:RANGE]).replace(" ", "").replace("[", "").replace("]", "").replace("\n", "")

    #chọn action:
    if(q_idx in per): #nếu trạng thái hiện tại đã có trong q_table
        #có xác suất epsilon thì khai phá ngẫu nhiên
        if np.random.rand() < epsilon:  # Thực hiện khai phá ngẫu nhiên
            act_idx = np.random.randint(0, len(arr_action))
        else:
            #láy ra action có điểm số cao nhất trong từ điển, ví dụ: {128: 0, 12: -760000.8} lấy act_idx = 128
            act_idx = max(per[q_idx], key=lambda k: per[q_idx][k])
            act_idx = np.where(arr_action == act_idx)[0][np.random.randint(0, len(np.where(arr_action == act_idx)[0]))]
    else: #nếu trạng thái hiện tại chưa có trong q_table thì tạo mới và chọn ngẫu nhiên
        
        per[q_idx] = make_new_q_value(p_state)
        act_idx = np.random.randint(0, len(arr_action))

    #hành động tiếp theo
    next_state = next_step(arr_action[act_idx], p_state)
    next_state_value = 0
    if (check_ended(next_state) == 2):
        next_state_value = TIE_REWARD
    else:
        if (check_ended(next_state) == -1):
            next_state_value = NOT_END
        else:
            if (check_ended(next_state) == p_id):
                next_state_value = WIN_REWARD
            else:
                next_state_value = LOSE_REWARD
            
    next_q_idx = str(next_state[:RANGE]).replace(" ", "").replace("[", "").replace("]", "").replace("\n", "")
    #nếu trạng thái tiếp theo chưa có trong q_table thì tạo mới
    if(next_q_idx not in per):
        per[next_q_idx] = make_new_q_value(next_state)
    
    #cập nhật điểm số cho trạng thái hiện tại
    per[q_idx][arr_action[act_idx]] = (1-LEARNING_RATE)*per[q_idx][arr_action[act_idx]] + LEARNING_RATE * (next_state_value + DISCOUNT_FACTOR * max(per[next_q_idx].values()))
    return arr_action[act_idx], per