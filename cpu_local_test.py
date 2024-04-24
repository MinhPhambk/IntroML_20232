from caro_cpu import cpu_run_n_game, cpu_run_one_game, cpu_bot_random, get_valid_actions, NUMBER_COLS, NUMBER_ROWS, next_step, check_ended

from q_cpu import make_random_q_table, q_bot_cpu
import time
import gc

#Chương trình test trên local
for p in range(1):
        #in ra số phần tử của per
    per = {}
    print("đọc file txt")
    start_time  = time.time()
    try:
        #đọc file per từ file txt
        with open('q_table' + NUMBER_COLS + '.txt', 'r') as f:
            for line in f:
                line = line.strip()
                key, value_str = line.split(':', 1)
                value_dict = eval(value_str)
                per[key] = value_dict
        print(type(per))
        if per is None:
            print("per is Nonbbe")
            per = make_random_q_table(10)
    except:
        print("per is None")
        per = make_random_q_table(10)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Đọc file txt thành công trong thời gian: ", execution_time)


    for i in range(100):

        print(len(per))
        win, per = cpu_run_one_game(q_bot_cpu, cpu_bot_random , per)
        if (win == 0):
            print("Your custom bot wins!")
        elif (win == 1):
            print("The random bot wins!")
        elif (win == 2):
            print("All tie!")

        start_time = time.time()
        print("start training....")
        win, per = cpu_run_n_game(q_bot_cpu, cpu_bot_random , per, 1000)
        # Lưu bảng Q vào tệp tin
        end_time = time.time()
        execution_time = end_time - start_time
        print("game đã xong, tiến hành lưu lại bảng Q: ")
        print("Your custom bot wins:", win[0])
        print("The random bot wins:", win[1])
        print("Trained time:", execution_time)


        # lưu bảng Q vào file txt
        print("Lưu bảng Q vào file txt")
        start_time = time.time()
        with open('q_table'+NUMBER_COLS+'.txt', 'w') as f:
            #ghi theo dạng key: value
            for key, value in per.items():
                f.write('%s:%s\n' % (key, value))
        end_time = time.time()
        execution_time = end_time - start_time
        print("Lưu file txt thành công trong thời gian: ", execution_time)
        print("kết thúc lươt lặp thứ: ", i)
        print("-------------------------------------------------\n")

        #chờ 5s
        time.sleep(5)
    print("kết thúc thế kỉ thứ: ", p)
    print("-------------------------------------------------\n")
    del per
    gc.collect()