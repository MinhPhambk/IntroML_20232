#khởi tạo 1 ma trận 15x15 ngẫu nhiên
import numpy as np
import math
from caro_cpu import get_valid_actions, next_step, check_ended, NUMBER_COLS, NUMBER_ROWS, next_step
from evaluate_for_q import evaluate
#khởi tạo điểm cho các trường hợp thắng, thua, hoà
WIN_REWARD = 100000000000000000000
LOSE_REWARD = -100000000000000000000
TIE_REWARD = NOT_END = 0
LEARNING_RATE = 0.8
DISCOUNT_FACTOR = 0.9
EPSILON = 0.21102003
RANGE = NUMBER_COLS * NUMBER_ROWS


q_table = {} # rỗng
def make_random_q_table(n):
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
    new_q_value_score = evaluate(p_state, p_id)
    #tạo ra dòng mới với điểm số cho từng action = điểm số của trạng thái mới * 1000000
    if new_q_value_score == math.inf:
        new_q_value[1] = WIN_REWARD
        return new_q_value
    elif new_q_value_score == -math.inf:
        new_q_value[1] = LOSE_REWARD
        return new_q_value
    for i in range(NUMBER_COLS * NUMBER_ROWS):
        if p_state[i] == 0:
            new_q_value[i] = new_q_value_score
        #NẾU dòng mới rỗng thì tạo ra 1 dòng mới với 1 action = 0
    if len(new_q_value) == 0:
        new_q_value[1] = 0
    return new_q_value


#q-learning sử dụng heuristic của Khải
def q_bot_cpu(p_state, per):

    q_idx = str(p_state[:RANGE]).replace(" ", "").replace("[", "").replace("]", "").replace("\n", "")
    arr_action = get_valid_actions(p_state)
    arr_action = np.where(arr_action == 1)[0]
    p_id = p_state[NUMBER_ROWS*NUMBER_COLS+2]%2
    epsilon = EPSILON

    #lượt của X
    if(not p_id):
        #chọn action:
        if(q_idx in per): #nếu trạng thái hiện tại đã có trong q_table
            #có xác suất epsilon thì khai phá ngẫu nhiên
            if np.random.rand() < epsilon:  # Thực hiện khai phá ngẫu nhiên
                act_idx = np.random.randint(0, len(arr_action))
            else:
                #láy ra action có điểm số cao nhất trong từ điển, ví dụ: {128: 0, 12: -760000.8} lấy act_idx = 128
                act_idx = max(per[q_idx], key=lambda k: per[q_idx][k])
                indexs = np.where(arr_action == act_idx)[0]
                act_idx = np.random.choice(indexs)
        else: #nếu trạng thái hiện tại chưa có trong q_table thì tạo mới và chọn ngẫu nhiên
            per[q_idx] = make_new_q_value(p_state)
            act_idx = np.random.randint(0, len(arr_action))

        #hành động tiếp theo
        next_state = next_step(arr_action[act_idx], p_state)
        next_state_value = evaluate(next_state, p_id)
        next_q_idx = str(next_state[:RANGE]).replace(" ", "").replace("[", "").replace("]", "").replace("\n", "")
        #nếu trạng thái tiếp theo chưa có trong q_table thì tạo mới
        if(next_q_idx not in per):
            per[next_q_idx] = make_new_q_value(next_state)

        #cập nhật điểm số cho trạng thái hiện tại
        print(q_idx,per[q_idx])
        print(next_q_idx,per[next_q_idx])
        per[q_idx][arr_action[act_idx]] = (1-LEARNING_RATE)*per[q_idx][arr_action[act_idx]] + LEARNING_RATE * (next_state_value + DISCOUNT_FACTOR * min(per[next_q_idx].values()))
    
    #lượt của O
    else:
        if(q_idx in per): #nếu trạng thái hiện tại đã có trong q_table
            #có xác suất epsilon thì khai phá ngẫu nhiên
            if np.random.rand() < epsilon:  # Thực hiện chọn hành động ngẫu nhiên trong q_table[q_idx]
                act_idx = np.random.randint(0, len(arr_action))
            else:
                #láy ra action có điểm số cao nhất trong từ điển, ví dụ: {128: 0, 12: -760000.8} lấy act_idx = 128
                act_idx = min(per[q_idx], key=lambda k: per[q_idx][k])
                indexs = np.where(arr_action == act_idx)[0]
                act_idx = np.random.choice(indexs)
        else: #nếu trạng thái hiện tại chưa có trong q_table thì tạo mới và chọn ngẫu nhiên
            per[q_idx] = make_new_q_value(p_state)
            act_idx = np.random.randint(0, len(arr_action))

        #hành động tiếp theo
        next_state = next_step(arr_action[act_idx], p_state)
        next_state_value = evaluate(next_state, p_id)
        next_q_idx = str(next_state[:RANGE]).replace(" ", "").replace("[", "").replace("]", "").replace("\n", "")

        #nếu trạng thái tiếp theo chưa có trong q_table thì tạo mới
        if(next_q_idx not in per):
            per[next_q_idx] = make_new_q_value(next_state)

        #cập nhật điểm số cho trạng thái hiện tại
        print (q_idx,per[q_idx])
        print (per[q_idx][arr_action[act_idx]])
        per[q_idx][arr_action[act_idx]] = (1-LEARNING_RATE)*per[q_idx][arr_action[act_idx]] + LEARNING_RATE * (next_state_value + DISCOUNT_FACTOR * max(per[next_q_idx].values()))


    return arr_action[act_idx], per


