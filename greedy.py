import math
import numpy as np
from caro_cpu import*

def evaluate(env_state,player):
    # Nếu hết cờ
    if check_ended(env_state)!=-1:
        # Trả về vô cùng nếu máy thắng
        if check_ended(env_state)==player%2:
            return math.inf
        elif check_ended(env_state)== (player+1)%2:
            return -math.inf # Trả về âm vô cùng nếu người thắng
        else:
            return 0 #tra ve 0 neu hoa
    # Tổng điểm của máy
    total_Score_Comp = 0
    # Tổng điểm của người
    total_Score_Human = 0

    # Chuỗi 4 ký tự của máy trên 5 ô liên tục không bị chặn 2 đầu nhưng vẫn có thể bị block ( Ví dụ _xx_xx_)
    # num4_Comp_Can_Block = 0
    # Chuỗi 4 ký tự của người trên 5 ô liên tục không bị chặn 2 đầu nhưng vẫn có thể bị block ( Ví dụ _xx_xx_)
    # num4_Human_Can_Block = 0

    # Chuỗi 4 ký tứ của máy trên 5 ô liên tục nhưng bị chặn 1 đầu ( Ví dụ xxx xo)
    num4_Comp_Block = 0
    # Chuỗi 4 ký tứ của người trên 5 ô liên tục nhưng bị chặn 1 đầu ( Ví dụ xxx xo)
    num4_Human_Block = 0

    # Chuỗi 4 ký tứ của người trên 5 ô liên tục không bị chặn cả 2 đầu ( Ví dụ o xxx x )
    num4_Human = 0
    # Chuỗi 4 ký tứ của máy trên 5 ô liên tục không bị chặn cả 2 đầu ( Ví dụ o xxx x )
    num4_Comp = 0

    # Chuỗi 3 ký tứ của người trên 5 ô liên tục không bị chặn cả 2 đầu ( Ví dụ o xx x )
    num3_Human = 0
    # Chuỗi 3 ký tứ của máy trên 5 ô liên tục không bị chặn cả 2 đầu ( Ví dụ o x xx )
    num3_Comp = 0

    # Chuỗi 3 ký tứ của người trên 5 ô liên tục nhưng bị chặn 1 đầu ( Ví dụ x x xo)
    num3_Human_Block = 0
    # Chuỗi 3 ký tứ của máy trên 5 ô liên tục nhưng bị chặn 1 đầu ( Ví dụ xx xo)
    num3_Comp_Block = 0

    # Chuỗi 2 ký tứ của người trên 5 ô liên tục không bị chặn cả 2 đầu ( Ví dụ o  x x )
    num2_Human = 0
    # Chuỗi 2 ký tứ của máy trên 5 ô liên tục không bị chặn cả 2 đầu ( Ví dụ o  x x )
    num2_Comp = 0

    # Chuỗi 2 ký tứ của người trên 5 ô liên tục nhưng bị chặn 1 đầu ( Ví dụ x xo)
    num2_Human_Block = 0
    # Chuỗi 2 ký tứ của máy trên 5 ô liên tục nhưng bị chặn 1 đầu ( Ví dụ x xo)
    num2_Comp_Block = 0

    # Số lượng các phần tử máy cạnh phần tử người
    near_By_Human = 0
    # Số lượng các phần tử người cạnh phần tử cạnh
    near_By_Comp = 0

    #mảng lưu các phần tử player đã đánh
    id_player= np.where(env_state[0:NUMBER_ACTIONS]==(player%2+1))[0]

    # Tính điểm cho Comp
    for id in id_player:
        xx,yy = convert_to_2D(id)
        
        # ĐẾM THEO HÀNG NGANG --

        # Kiểm tra xem tọa độ của điểm có trong trường hợp "tệ" hay không
        ## Trường hợp 1 là đứng đằng trước nó đã có 1 điểm khác được xét rồi (sẽ bị trùng dữ liệu, dữ liệu trùng thậm chí không đem lại tác dụng gì)
        ## Trường hợp 2 là 2 ô liên tiếp phía trước nó không có chứa ký tự gì cả
        ### Cách giải thích trên sẽ xuyên suốt hàm heuristic!
        b = False
        if id%NUMBER_COLS!=0 and env_state[id-1]== player%2+1:
            b = True
        if not b and (id)%NUMBER_COLS<NUMBER_COLS-2 and env_state[id+1]==0 and  env_state[id+2]==0 :
            b = True

        # Kiểm tra xem đằng trước nó có phải cạnh của bàn cờ hoặc một ký tự của người hay không
        ## Nếu có thì những chuỗi sẽ xét tới đều nằm trong diện "bị chặn"
        ### Cách giải thích này cũng tổng quát với cả 4 đường ngang, thẳng, C1, C2
        a = False
        if id%NUMBER_COLS>0 and env_state[id-1]== (player+1)%2+1:
            a = True
        if id%NUMBER_COLS==0:
            a = True
        # nSpace là số ký tự trắng trên 4 phần tử đằng sau phần tử đang xét
        nSpace = 0

        # Duyệt 4 phần tử đằng sau nó
        for i in range(1, 5):
            # Nếu tệ thì hủy duyệt
            if b:
                break
            # Nếu gặp phải ký tự của người
            if id%NUMBER_COLS<NUMBER_COLS-i and env_state[id+i]== (player+1)%2+1:
                # Nếu bị chặn thì sẽ loại (chặn 2 đầu không thể giành chiến thắng)
                if a:
                    break
                else:
                    # Cấp nhật số lượng các chuỗi vừa định nghĩa
                    ## Nếu đằng trước ký tự người là ký tự máy thì chuỗi sẽ bị chặn 1 đầu ( __xx_xo )
                    if env_state[id+i-1] != 0:
                        if i - nSpace - 1 == 1:
                            num2_Comp_Block += 1
                        elif i - nSpace - 1 == 2:
                            # Trường hợp luôn tạo thành num3_Bloc
                            ## _x_xxo
                            if nSpace == 1:
                                num3_Comp_Block += 1
                            # Trường hợp chỉ tạo thành num3_Block điều kiện đặc biệt
                            ## Ví dụ __xxxo (tạo thành),, o_xxxo (loại)
                            elif (id-2)%NUMBER_COLS>=0 and env_state[id-2] == 0:
                                num3_Comp_Block += 1
                        elif i - nSpace - 1 == 3:
                            num4_Comp_Block += 1
                    ## Nếu không phải ký tự máy thì không bị chặn đầu nào ( _xxx_o__)
                    else:
                        if i - nSpace - 1 == 1:
                            num2_Comp += 1
                        elif i - nSpace - 1 == 2:
                            # Trường hợp tốt: num3_Comp có thể mở rộng trực tiếp thành num4_Comp: __xxx_o
                            if (id-2)%NUMBER_COLS>=0 and  env_state[id-2] == 0:
                                num3_Comp += 1
                            else:
                                num3_Comp_Block += 1 # num3 chỉ có thể mở rộng lên num4_Block: |_xxx_o hoặc o_xxx_o
                    break

            # Nếu gặp phần tử trắng thì cập nhật số lượng
            if id%NUMBER_COLS+i<NUMBER_COLS and  env_state[id+i] == 0:
                nSpace += 1
            
            # Sau khi duyện hết 4 phần tử, cập nhật số lượng các chuỗi
            if i == 4 and id%NUMBER_COLS+i<NUMBER_COLS:
                # Trường hợp xấu nhất nếu phần tử thứ 4 rơi vào cột cuối cùng của bảng và có ký tự máy ( Ví dụ __xx_xx| ) -> Bị chặn 1 đầu
                if id%NUMBER_COLS+i== NUMBER_COLS - 1 and  env_state[id+i] == player%2+1:
                    if i - nSpace == 3:
                        num4_Comp_Block += 1
                    elif i - nSpace == 2:
                        num3_Comp_Block += 1
                    elif i - nSpace == 1:
                        num2_Comp_Block += 1

                else:
                    if i - nSpace == 1:
                        if a:
                            num2_Comp_Block += 1
                        else:
                            num2_Comp += 1
                    elif i - nSpace == 2:
                        # Trường hợp đặc biệt, kể cả có ký tự người ở trước hay không thì 3 ký tự trong 5 ô liên tiếp không thể chuyển thành num4_Comp
                        ## Ví dụ: __x_x_x__
                        if env_state[id+i] == player%2+1:
                            num3_Comp_Block += 1
                        else:
                            if a:
                                num3_Comp_Block += 1
                            else: 
                                num3_Comp += 1
                    elif i - nSpace == 3:
                        # Trường hợp đặc biệt, kể cả có ký tự người ở trước hay không thì 4 ký tự máy trong 5 ô liên tiếp vẫn có thể bị block
                        ## Ví dụ __xx_xx___ -> __xxoxx___
                        if env_state[id+i] == player%2+1:
                            num4_Comp_Block += 1
                        else:
                            if a:
                                num4_Comp_Block += 1
                            else:
                                num4_Comp += 1
            # Trường hợp mở rộng ra 4 ô đằng sau vượt quá ranh giới trò chơi ( chuỗi sẽ tự động bị chặn 1 đầu ở ranh giới )
            if id%NUMBER_COLS+i>=NUMBER_COLS:
                # Nếu bị chặn nốt đầu còn lại thì thoát ( __oxxxx| )
                if a:
                    break
                else:
                    if i - nSpace - 1 == 1:
                        num2_Comp_Block += 1
                    elif i - nSpace - 1 == 2:
                        # Trường hợp đặc biệt: phần tử cuối cùng trước khi chạm thành bảng là phần tử trống -> chuỗi 3 liên tục có thể mở rộng thành num4
                        ## Ví dụ: __xxx_| -> _xxxx_|
                        if env_state[id+i-1] == 0:
                            # Trường hợp tạo thành num 3 trong điều kiện đặc biệt
                            ## Ví dụ __xxx_|
                            if id%NUMBER_COLS - 2 >= 0 and env_state[id-2] == 0:
                                num3_Comp += 1
                            ## Không tạo được thành num 3: o_xxx_|
                            else:
                                num3_Comp_Block += 1
                        else:
                            # Trường hợp nếu có khoảng trắng -> luôn luôn tạo num3_Block
                            ## Ví dụ o_xx_x|
                            if nSpace == 1:
                                num3_Comp_Block += 1
                            # Nếu không có khoảng trắng thì tạo num3_Block trong điều kiện đặc biệt
                            ## Ví dụ: __xxx| (tạo thành) ,, o_xxx| (loại)
                            elif id%NUMBER_COLS - 2 >= 0 and  env_state[id-2] == 0:
                                num3_Comp_Block += 1
                    elif i - nSpace - 1== 3: # Duy nhất trường hợp: __xxxx|
                        num4_Comp_Block += 1
                    break


        # ĐẾM THEO HÀNG DỌC |
        b = False
        if xx > 0 and env_state[id-NUMBER_COLS]== player%2+1:
            b = True
        if not b and xx + 2 < NUMBER_ROWS and env_state[id+NUMBER_COLS]== 0 and env_state[id+2*NUMBER_COLS]== 0:
            b = True

        a = False
        nSpace = 0
        if xx > 0 and env_state[id-NUMBER_COLS]== (player+1)%2+1:
            a = True
        if xx == 0:
            a = True

        for i in range(1, 5):
            if b:
                break
            if xx + i < NUMBER_ROWS and env_state[id+i*NUMBER_COLS]== (player+1)%2+1:
                if a:
                    break
                else:
                    if env_state[id+(i-1)*NUMBER_COLS] != 0:
                        if i - nSpace - 1 == 1:
                            num2_Comp_Block += 1
                        elif i - nSpace - 1 == 2:
                            if nSpace == 1:
                                num3_Comp_Block += 1
                            elif xx - 2 >= 0 and env_state[id-2*NUMBER_COLS] == 0:
                                num3_Comp_Block += 1
                        elif i - nSpace - 1 == 3:
                            num4_Comp_Block += 1
                    else:
                        if i - nSpace - 1 == 1:
                            num2_Comp += 1
                        elif i - nSpace - 1 == 2:
                            if xx - 2 >= 0 and env_state[id-2*NUMBER_COLS] == 0:
                                num3_Comp += 1
                            else:
                                num3_Comp_Block += 1
                    break
            if xx + i < NUMBER_ROWS and env_state[id+i*NUMBER_COLS] == 0:
                nSpace += 1
            if i == 4 and xx + i < NUMBER_ROWS:
                if xx + i == NUMBER_ROWS - 1 and env_state[id+i*NUMBER_COLS] == player%2+1:
                    if i - nSpace == 3:
                        num4_Comp_Block += 1
                    elif i - nSpace == 2:
                        num3_Comp_Block += 1
                    elif i - nSpace == 1:
                        num2_Comp_Block += 1

                else:
                    if i - nSpace == 1:
                        if a:
                            num2_Comp_Block += 1
                        else:
                            num2_Comp += 1
                    elif i - nSpace == 2:
                        if env_state[id+i*NUMBER_COLS] == player%2+1:
                            num3_Comp_Block += 1
                        else:
                            if a:
                                num3_Comp_Block += 1
                            else:
                                num3_Comp += 1
                    elif i - nSpace == 3:
                        if env_state[id+i*NUMBER_COLS] == player%2+1:
                            num4_Comp_Block += 1
                        else:
                            if a:
                                num4_Comp_Block += 1
                            else:
                                num4_Comp += 1
            if xx + i >= NUMBER_ROWS:
                if a:
                    break
                else:
                    if i - nSpace - 1 == 1:
                        num2_Comp_Block += 1
                    elif i - nSpace - 1 == 2:
                        if env_state[id+(i-1)*NUMBER_COLS] == 0:
                            if xx - 2 >= 0 and env_state[id-2*NUMBER_COLS] == 0:
                                num3_Comp += 1
                            else:
                                num3_Comp_Block += 1
                        else:
                            if nSpace == 1:
                                num3_Comp_Block += 1
                            elif xx - 2 >= 0 and env_state[id-2*NUMBER_COLS] == 0:
                                num3_Comp_Block += 1
                    elif i - nSpace -1 == 3:
                        num4_Comp_Block += 1
                    break
        
        # ĐẾM THEO ĐƯỜNG CHÉO C1 \
        b = False
        if yy > 0 and xx > 0 and env_state[id-1-NUMBER_COLS] == player%2+1:
            b = True
        if not b and yy + 2 < NUMBER_COLS and xx + 2 < NUMBER_ROWS and  env_state[id+1+NUMBER_COLS] == 0 and  env_state[id+2+2*NUMBER_COLS] == 0:
            b = True

        a = False
        nSpace = 0
        if yy > 0 and xx > 0 and  env_state[id-1-NUMBER_COLS] == (player+1)%2+1:
            a = True
        if yy == 0 or xx == 0:
            a = True

        for i in range(1, 5):
            if b:
                break
            if yy + i < NUMBER_COLS and xx + i < NUMBER_ROWS and env_state[id+i+i*NUMBER_COLS] == (player+1)%2+1:
                if a:
                    break
                else:
                    if env_state[id+(i-1)+(i-1)*NUMBER_COLS] != 0:
                        if i - nSpace - 1 == 1:
                            num2_Comp_Block += 1
                        elif i - nSpace - 1 == 2:
                            if nSpace == 1:
                                num3_Comp_Block += 1
                            elif xx - 2 >= 0 and yy - 2 >= 0 and env_state[id-2-2*NUMBER_COLS] == 0:
                                num3_Comp_Block += 1
                        elif i - nSpace - 1 == 3:
                            num4_Comp_Block += 1
                    else:
                        if i - nSpace - 1 == 1:
                            num2_Comp += 1
                        elif i - nSpace - 1 == 2:
                            if xx - 2 >= 0 and yy - 2 >= 0 and env_state[id-2-2*NUMBER_COLS] == 0:
                                num3_Comp += 1
                            else:
                                num3_Comp_Block += 1
                    break
            if yy + i < NUMBER_COLS and xx + i < NUMBER_ROWS and env_state[id+i+i*NUMBER_COLS] == 0:
                nSpace += 1
            if i == 4 and yy + i < NUMBER_COLS and xx + i < NUMBER_ROWS:
                if (yy + i == NUMBER_COLS - 1 or xx + i == NUMBER_ROWS - 1) and env_state[id+i+i*NUMBER_COLS] == (player)%2+1:
                    if i - nSpace == 3:
                        num4_Comp_Block += 1
                    elif i - nSpace == 2:
                        num3_Comp_Block += 1
                    elif i - nSpace == 1:
                        num2_Comp_Block += 1

                else:
                    if i - nSpace == 1:
                        if a:
                            num2_Comp_Block += 1
                        else:
                            num2_Comp += 1
                    elif i - nSpace == 2:
                        if env_state[id+i+i*NUMBER_COLS] == (player)%2+1:
                            num3_Comp_Block += 1
                        else:
                            if a:
                                num3_Comp_Block += 1
                            else:
                                num3_Comp += 1
                    elif i - nSpace == 3:
                        if env_state[id+i+i*NUMBER_COLS] == (player)%2+1:
                            num4_Comp_Block += 1
                        else:
                            if a:
                                num4_Comp_Block += 1
                            else:
                                num4_Comp += 1
            if xx + i >= NUMBER_ROWS or yy + i >= NUMBER_COLS:
                if a:
                    break
                else:
                    if i - nSpace - 1 == 1:
                        num2_Comp_Block += 1
                    elif i - nSpace - 1 == 2:
                        if  env_state[id+(i-1)+(i-1)*NUMBER_COLS] == 0:
                            if xx - 2 >= 0 and yy - 2 >= 0 and env_state[id-2-2*NUMBER_COLS] == 0:
                                num3_Comp += 1
                            else:
                                num3_Comp_Block += 1
                        else:
                            if nSpace == 1:
                                num3_Comp_Block += 1
                            elif xx - 2 >= 0 and yy - 2 >= 0 and env_state[id-2-2*NUMBER_COLS] == 0:
                                num3_Comp_Block += 1
                    elif i - nSpace - 1 == 3:
                        num4_Comp_Block += 1
                    break
        
        # ĐẾM THEO ĐƯỜNG CHÉO C2 /
        b = False
        if yy + 1 < NUMBER_COLS and xx > 0 and env_state[id+1-1*NUMBER_COLS] == (player)%2+1:
            b = True
        if not b and yy - 2 >= 0 and xx + 2 < NUMBER_ROWS  and env_state[id-1+1*NUMBER_COLS] == 0 and env_state[id-2+2*NUMBER_COLS] == 0:
            b = True

        a = False
        nSpace = 0
        if yy < NUMBER_COLS - 1 and xx > 0 and env_state[id+1-1*NUMBER_COLS] == (player+1)%2+1:
            a = True
        if yy == NUMBER_COLS - 1 or xx == 0:
            a = True

        for i in range(1, 5):
            if b:
                break
            if yy - i >= 0 and xx + i < NUMBER_ROWS and env_state[id-i+i*NUMBER_COLS] == (player+1)%2+1:
                if a:
                    break
                else:
                    if env_state[id-(i-1)+(i-1)*NUMBER_COLS] != 0:
                        if i - nSpace - 1 == 1:
                            num2_Comp_Block += 1
                        elif i - nSpace - 1 == 2:
                            if nSpace == 1:
                                num3_Comp_Block += 1
                            elif xx - 2 >= 0 and yy + 2 < NUMBER_COLS and env_state[id+2-2*NUMBER_COLS] == 0:
                                num3_Comp_Block += 1
                        elif i - nSpace - 1 == 3:
                            num4_Comp_Block += 1
                    else:
                        if i - nSpace - 1 == 1:
                            num2_Comp += 1
                        elif i - nSpace - 1 == 2:
                            if yy + 2 < NUMBER_COLS and xx - 2 >= 0 and env_state[id+2-2*NUMBER_COLS] == 0:
                                num3_Comp += 1
                            else:
                                num3_Comp_Block += 1
                    break
            if yy - i >= 0  and xx + i < NUMBER_ROWS and env_state[id-i+i*NUMBER_COLS] == 0:
                nSpace += 1
            if i == 4 and yy - i >= 0  and xx + i < NUMBER_ROWS:
                if (yy - i == 0 or xx + i == NUMBER_ROWS - 1) and env_state[id-i+i*NUMBER_COLS] == player%2+1:
                    if i - nSpace == 3:
                        num4_Comp_Block += 1
                    elif i - nSpace == 2:
                        num3_Comp_Block += 1
                    elif i - nSpace == 1:
                        num2_Comp_Block += 1

                else:
                    if i - nSpace == 1:
                        if a:
                            num2_Comp_Block += 1
                        else:
                            num2_Comp += 1
                    elif i - nSpace == 2:
                        if env_state[id-i+i*NUMBER_COLS] == player%2+1:
                            num3_Comp_Block += 1
                        else:
                            if a:
                                num3_Comp_Block += 1
                            else:
                                num3_Comp += 1
                    elif i - nSpace == 3:
                        if env_state[id-i+i*NUMBER_COLS] == player%2+1:
                            num4_Comp_Block += 1
                        else:
                            if a:
                                num4_Comp_Block += 1
                            else:
                                num4_Comp += 1
            if xx + i >= NUMBER_ROWS or yy - i < 0:
                if a:
                    break
                else:
                    if i - nSpace - 1== 1:
                        num2_Comp_Block += 1
                    elif i - nSpace - 1== 2:
                        if env_state[id-(i-1)+(i-1)*NUMBER_COLS] == 0:
                            if xx - 2 >= 0 and yy + 2 < NUMBER_COLS and env_state[id+2-2*NUMBER_COLS] == 0:
                                num3_Comp += 1
                            else:
                                num3_Comp_Block += 1
                        else:
                            if nSpace == 1:
                                num3_Comp_Block += 1
                            elif xx - 2 >= 0 and yy + 2 < NUMBER_COLS and env_state[id+2-2*NUMBER_COLS] == 0:
                                num3_Comp_Block += 1
                    elif i - nSpace - 1== 3:
                        num4_Comp_Block += 1
                    break
        
        if xx + 1 < NUMBER_ROWS and env_state[id+1*NUMBER_COLS] == (player+1)%2+1:
            near_By_Comp += 1
        if yy + 1 < NUMBER_COLS and env_state[id+1] == (player+1)%2+1:
            near_By_Comp += 1
        if xx > 0 and env_state[id-1*NUMBER_COLS] == (player+1)%2+1:
            near_By_Comp += 1
        if yy > 0 and env_state[id-1] == (player+1)%2+1:
            near_By_Comp += 1
        if xx + 1 < NUMBER_ROWS and yy + 1 < NUMBER_COLS and env_state[id+1+1*NUMBER_COLS] == (player+1)%2+1:
            near_By_Comp += 1
        if xx + 1 < NUMBER_ROWS and yy > 0 and env_state[id-1+1*NUMBER_COLS] == (player+1)%2+1:
            near_By_Comp += 1
        if yy + 1 < NUMBER_COLS and xx > 0 and env_state[id+1-1*NUMBER_COLS] == (player+1)%2+1:
            near_By_Comp += 1
        if yy > 0 and xx > 0 and env_state[id-1-1*NUMBER_COLS] == (player+1)%2+1:
            near_By_Comp += 1

    # Tính điểm cho Người

    id_enermy= np.where(env_state[0:NUMBER_ACTIONS]==((player+1)%2+1))[0]
    for id in id_enermy:
        xx,yy= convert_to_2D(id)
        
        # ĐẾM THEO HÀNG NGANG --
        b = False
        if yy > 0 and env_state[id-1]== (player+1)%2+1:
            b = True
        if not b and yy + 2 < NUMBER_COLS and env_state[id+1]==0 and  env_state[id+2]==0 :
            b = True

        a = False
        if yy > 0 and env_state[id-1]== (player)%2+1:
            a = True
        if yy == 0:
            a = True
        nSpace = 0

        for i in range(1, 5):
            if b:
                break
            if yy + i < NUMBER_COLS and env_state[id+i]== (player)%2+1:
                if a:
                    break
                else:
                    if env_state[id+i-1] != 0:
                        if i - nSpace - 1 == 1:
                            num2_Human_Block += 1
                        elif i - nSpace - 1 == 2:
                            if nSpace == 1:
                                num3_Human_Block += 1
                            elif yy - 2 >= 0 and env_state[id-2] == 0:
                                num3_Human_Block += 1
                        elif i - nSpace - 1 == 3:
                            num4_Human_Block += 1
                    else:
                        if i - nSpace - 1 == 1:
                            num2_Human += 1
                        elif i - nSpace - 1 == 2:
                            if yy - 2 >= 0 and  env_state[id-2] == 0:
                                num3_Human += 1
                            else:
                                num3_Human_Block += 1 
                    break
            if yy + i < NUMBER_COLS and env_state[id+i] == 0:
                nSpace += 1
            
            if i == 4 and yy + i < NUMBER_COLS:
                if yy + i == NUMBER_COLS - 1 and env_state[id+i] == (player+1)%2+1:
                    if i - nSpace == 3:
                        num4_Human_Block += 1
                    elif i - nSpace == 2:
                        num3_Human_Block += 1
                    elif i - nSpace == 1:
                        num2_Human_Block += 1

                else:
                    if i - nSpace == 1:
                        if a:
                            num2_Human_Block += 1
                        else:
                            num2_Human += 1
                    elif i - nSpace == 2:
                        if env_state[id+i] == (player+1)%2+1:
                            num3_Human_Block += 1
                        else:
                            if a:
                                num3_Human_Block += 1
                            else: 
                                num3_Human += 1
                    elif i - nSpace == 3:
                        if env_state[id+i] == (player+1)%2+1:
                            num4_Human_Block += 1
                        else:
                            if a:
                                num4_Human_Block += 1
                            else:
                                num4_Human += 1
            if yy + i >= NUMBER_COLS:
                if a:
                    break
                else:
                    if i - nSpace - 1 == 1:
                        num2_Human_Block += 1
                    elif i - nSpace - 1 == 2:
                        if env_state[id+i-1] == 0:
                            if yy - 2 >= 0 and env_state[id-2] == 0:
                                num3_Human += 1
                            else:
                                num3_Human_Block += 1
                        else:
                            if nSpace == 1:
                                num3_Human_Block += 1
                            elif yy - 2 >= 0 and env_state[id-2] == 0:
                                num3_Human_Block += 1
                    elif i - nSpace - 1== 3: 
                        num4_Human_Block += 1
                    break


        # ĐẾM THEO HÀNG DỌC |
        b = False
        if xx > 0 and env_state[id-NUMBER_COLS]== (player+1)%2+1:
            b = True
        if not b and xx + 2 < NUMBER_ROWS and env_state[id+NUMBER_COLS]== 0 and env_state[id+2*NUMBER_COLS]== 0:
            b = True

        a = False
        nSpace = 0
        if xx > 0 and env_state[id-NUMBER_COLS]== (player)%2+1:
            a = True
        if xx == 0:
            a = True

        for i in range(1, 5):
            if b:
                break
            if xx + i < NUMBER_ROWS and env_state[id+i*NUMBER_COLS]== (player)%2+1:
                if a:
                    break
                else:
                    if env_state[id+(i-1)*NUMBER_COLS] != 0:
                        if i - nSpace - 1 == 1:
                            num2_Human_Block += 1
                        elif i - nSpace - 1 == 2:
                            if nSpace == 1:
                                num3_Human_Block += 1
                            elif xx - 2 >= 0 and env_state[id-2*NUMBER_COLS] == 0:
                                num3_Human_Block += 1
                        elif i - nSpace - 1 == 3:
                            num4_Human_Block += 1
                    else:
                        if i - nSpace - 1 == 1:
                            num2_Human += 1
                        elif i - nSpace - 1 == 2:
                            if xx - 2 >= 0 and env_state[id-2*NUMBER_COLS] == 0:
                                num3_Human += 1
                            else:
                                num3_Human_Block += 1
                    break
            if xx + i < NUMBER_ROWS and env_state[id+i*NUMBER_COLS] == 0:
                nSpace += 1
            if i == 4 and xx + i < NUMBER_ROWS:
                if xx + i == NUMBER_ROWS - 1 and env_state[id+i*NUMBER_COLS] == (player+1)%2+1:
                    if i - nSpace == 3:
                        num4_Human_Block += 1
                    elif i - nSpace == 2:
                        num3_Human_Block += 1
                    elif i - nSpace == 1:
                        num2_Human_Block += 1

                else:
                    if i - nSpace == 1:
                        if a:
                            num2_Human_Block += 1
                        else:
                            num2_Human += 1
                    elif i - nSpace == 2:
                        if env_state[id+i*NUMBER_COLS] == (player+1)%2+1:
                            num3_Human_Block += 1
                        else:
                            if a:
                                num3_Human_Block += 1
                            else:
                                num3_Human += 1
                    elif i - nSpace == 3:
                        if env_state[id+i*NUMBER_COLS] == (player+1)%2+1:
                            num4_Human_Block += 1
                        else:
                            if a:
                                num4_Human_Block += 1
                            else:
                                num4_Human += 1
            if xx + i >= NUMBER_ROWS:
                if a:
                    break
                else:
                    if i - nSpace - 1 == 1:
                        num2_Human_Block += 1
                    elif i - nSpace - 1 == 2:
                        if env_state[id+(i-1)*NUMBER_COLS] == 0:
                            if xx - 2 >= 0 and env_state[id-2*NUMBER_COLS] == 0:
                                num3_Human += 1
                            else:
                                num3_Human_Block += 1
                        else:
                            if nSpace == 1:
                                num3_Human_Block += 1
                            elif xx - 2 >= 0 and env_state[id-2*NUMBER_COLS] == 0:
                                num3_Human_Block += 1
                    elif i - nSpace -1 == 3:
                        num4_Human_Block += 1
                    break
        
        # ĐẾM THEO ĐƯỜNG CHÉO C1 \
        b = False
        if yy > 0 and xx > 0 and env_state[id-1-NUMBER_COLS] == (player+1)%2+1:
            b = True
        if not b and yy + 2 < NUMBER_COLS and xx + 2 < NUMBER_ROWS and env_state[id+1+NUMBER_COLS] == 0 and  env_state[id+2+2*NUMBER_COLS] == 0:
            b = True

        a = False
        nSpace = 0
        if yy > 0 and xx > 0 and  env_state[id-1-NUMBER_COLS] == (player)%2+1:
            a = True
        if yy == 0 or xx == 0:
            a = True

        for i in range(1, 5):
            if b:
                break
            if yy + i < NUMBER_COLS and xx + i < NUMBER_ROWS and env_state[id+i+i*NUMBER_COLS] == (player)%2+1:
                if a:
                    break
                else:
                    if env_state[id+(i-1)+(i-1)*NUMBER_COLS] != 0:
                        if i - nSpace - 1 == 1:
                            num2_Human_Block += 1
                        elif i - nSpace - 1 == 2:
                            if nSpace == 1:
                                num3_Human_Block += 1
                            elif xx - 2 >= 0 and yy - 2 >= 0 and env_state[id-2-2*NUMBER_COLS] == 0:
                                num3_Human_Block += 1
                        elif i - nSpace - 1 == 3:
                            num4_Human_Block += 1
                    else:
                        if i - nSpace - 1 == 1:
                            num2_Human += 1
                        elif i - nSpace - 1 == 2:
                            if xx - 2 >= 0 and yy - 2 >= 0 and env_state[id-2-2*NUMBER_COLS] == 0:
                                num3_Human += 1
                            else:
                                num3_Human_Block += 1
                    break
            if yy + i < NUMBER_COLS and xx + i <NUMBER_ROWS and env_state[id+i+i*NUMBER_COLS] == 0:
                nSpace += 1
            if i == 4 and yy + i < NUMBER_COLS and xx + i < NUMBER_ROWS:
                if (yy + i == NUMBER_COLS - 1 or xx + i == NUMBER_ROWS - 1) and env_state[id+i+i*NUMBER_COLS] == (player+1)%2+1:
                    if i - nSpace == 3:
                        num4_Human_Block += 1
                    elif i - nSpace == 2:
                        num3_Human_Block += 1
                    elif i - nSpace == 1:
                        num2_Human_Block += 1

                else:
                    if i - nSpace == 1:
                        if a:
                            num2_Human_Block += 1
                        else:
                            num2_Human += 1
                    elif i - nSpace == 2:
                        if  env_state[id+i+i*NUMBER_COLS] == (player+1)%2+1:
                            num3_Human_Block += 1
                        else:
                            if a:
                                num3_Human_Block += 1
                            else:
                                num3_Human += 1
                    elif i - nSpace == 3:
                        if env_state[id+i+i*NUMBER_COLS] == (player+1)%2+1:
                            num4_Human_Block += 1
                        else:
                            if a:
                                num4_Human_Block += 1
                            else:
                                num4_Human += 1
            if xx + i >= NUMBER_ROWS or yy + i >= NUMBER_COLS:
                if a:
                    break
                else:
                    if i - nSpace - 1 == 1:
                        num2_Human_Block += 1
                    elif i - nSpace - 1 == 2:
                        if env_state[id+(i-1)+(i-1)*NUMBER_COLS] == 0:
                            if xx - 2 >= 0 and yy - 2 >= 0 and env_state[id-2-2*NUMBER_COLS] == 0:
                                num3_Human += 1
                            else:
                                num3_Human_Block += 1
                        else:
                            if nSpace == 1:
                                num3_Human_Block += 1
                            elif xx - 2 >= 0 and yy - 2 >= 0 and env_state[id-2-2*NUMBER_COLS] == 0:
                                num3_Human_Block += 1
                    elif i - nSpace - 1 == 3:
                        num4_Human_Block += 1
                    break
        
        # ĐẾM THEO ĐƯỜNG CHÉO C2 /
        b = False
        if yy + 1 < NUMBER_COLS and xx > 0 and env_state[id+1-1*NUMBER_COLS] == (player+1)%2+1:
            b = True
        if not b and yy - 2 >= 0 and xx + 2 < NUMBER_ROWS  and env_state[id-1+1*NUMBER_COLS] == 0 and env_state[id-2+2*NUMBER_COLS] == 0:
            b = True

        a = False
        nSpace = 0
        if yy < NUMBER_COLS - 1 and xx > 0 and env_state[id+1-1*NUMBER_COLS] == (player)%2+1:
            a = True
        if yy == NUMBER_COLS - 1 or xx == 0:
            a = True

        for i in range(1, 5):
            if b:
                break
            if yy - i >= 0 and xx + i < NUMBER_ROWS and env_state[id-i+i*NUMBER_COLS] == (player)%2+1:
                if a:
                    break
                else:
                    if env_state[id-(i-1)+(i-1)*NUMBER_COLS] != 0:
                        if i - nSpace - 1 == 1:
                            num2_Human_Block += 1
                        elif i - nSpace - 1 == 2:
                            if nSpace == 1:
                                num3_Human_Block += 1
                            elif xx - 2 >= 0 and yy + 2 < NUMBER_COLS and env_state[id+2-2*NUMBER_COLS] == 0:
                                num3_Human_Block += 1
                        elif i - nSpace - 1 == 3:
                            num4_Human_Block += 1
                    else:
                        if i - nSpace - 1 == 1:
                            num2_Human += 1
                        elif i - nSpace - 1 == 2:
                            if yy + 2 < NUMBER_COLS and xx - 2 >= 0 and env_state[id+2-2*NUMBER_COLS] == 0:
                                num3_Human += 1
                            else:
                                num3_Human_Block += 1
                    break
            if yy - i >= 0  and xx + i < NUMBER_ROWS and  env_state[id-i+i*NUMBER_COLS] == 0:
                nSpace += 1
            if i == 4 and yy - i >= 0  and xx + i < NUMBER_ROWS:
                if (yy - i == 0 or xx + i == NUMBER_ROWS - 1) and env_state[id-i+i*NUMBER_COLS] == (player+1)%2+1:
                    if i - nSpace == 3:
                        num4_Human_Block += 1
                    elif i - nSpace == 2:
                        num3_Human_Block += 1
                    elif i - nSpace == 1:
                        num2_Human_Block += 1

                else:
                    if i - nSpace == 1:
                        if a:
                            num2_Human_Block += 1
                        else:
                            num2_Human += 1
                    elif i - nSpace == 2:
                        if env_state[id-i+i*NUMBER_COLS] == (player+1)%2+1:
                            num3_Human_Block += 1
                        else:
                            if a:
                                num3_Human_Block += 1
                            else:
                                num3_Human += 1
                    elif i - nSpace == 3:
                        if env_state[id-i+i*NUMBER_COLS] == (player+1)%2+1:
                            num4_Human_Block += 1
                        else:
                            if a:
                                num4_Human_Block += 1
                            else:
                                num4_Human += 1
            if xx + i >= NUMBER_ROWS or yy - i < 0:
                if a:
                    break
                else:
                    if i - nSpace - 1== 1:
                        num2_Human_Block += 1
                    elif i - nSpace - 1== 2:
                        if  env_state[id-(i-1)+(i-1)*NUMBER_COLS] == 0:
                            if xx - 2 >= 0 and yy + 2 < NUMBER_COLS and env_state[id+2-2*NUMBER_COLS] == 0:
                                num3_Human += 1
                            else:
                                num3_Human_Block += 1
                        else:
                            if nSpace == 1:
                                num3_Human_Block += 1
                            elif xx - 2 >= 0 and yy + 2 < NUMBER_COLS and env_state[id+2-2*NUMBER_COLS] == 0:
                                num3_Human_Block += 1
                    elif i - nSpace - 1== 3:
                        num4_Human_Block += 1
                    break
        
        if xx + 1 < NUMBER_ROWS and env_state[id+1*NUMBER_COLS] == (player)%2+1:
            near_By_Human += 1
        if yy + 1 < NUMBER_COLS and env_state[id+1] == (player)%2+1:
            near_By_Human += 1
        if xx > 0 and env_state[id-1*NUMBER_COLS] == (player)%2+1:
            near_By_Human += 1
        if yy > 0 and env_state[id-1] == (player)%2+1:
            near_By_Human += 1
        if xx + 1 < NUMBER_ROWS and yy + 1 < NUMBER_COLS and env_state[id+1+1*NUMBER_COLS] == (player)%2+1:
            near_By_Human += 1
        if xx + 1 < NUMBER_ROWS and yy > 0 and env_state[id-1+1*NUMBER_COLS] == (player)%2+1:
            near_By_Human += 1
        if yy + 1 < NUMBER_COLS and xx > 0 and  env_state[id+1-1*NUMBER_COLS] == (player)%2+1:
            near_By_Human += 1
        if yy > 0 and xx > 0 and env_state[id-1-1*NUMBER_COLS] == (player)%2+1:
            near_By_Human += 1
    

    # Công thức tính điểm bàn cờ của hàm heuristic h(n)

    turn = env_state[NUMBER_ACTIONS+2]
    ## Trong trường hợp player là người, tức nước đi vừa rồi là của máy đánh
    if turn%2 == player:
        # Nếu máy đánh xong mà bàn cờ vẫn còn 4 ký tự người liên tục ( dù bị block hay không ) thì người thắng
        if num4_Human > 0 or num4_Human_Block > 0:
            return -5555555555555555555
        # Nếu máy đánh được nước dẫn tới 4 ký tự máy liên tục không bị chặn thì máy thắng
        if num4_Comp > 0:
            return 5555555555555555555
        # Nếu máy đánh được nước dẫn tới nước đôi 4 ký tự máy liên tục bị chặn thì máy thắng
        if num4_Comp_Block >= 2:
            return 5555555555555555555
        # Nếu máy không có nước 4 nào và người có nước 3 không bị chặn thì người thắng
        ## Giải thích: tới phiên của người sẽ đánh biến nước 3 thành nước 4 không bị chặn, khi ấy người sẽ thắng
        if num4_Comp_Block == 0 and num3_Human > 0:
            return -555555555555555555
        # Nếu máy có nước đôi 3 ký tự không bị chặn trong khi đó người không có nước 3 nào thì máy sẽ thắng
        ## Trong trường hợp người có nước 3 chưa chắc máy đã thắng
        ### Ví dụ: xxx          ->       xxx        ->         xxx        ->       oxxx
                #    x                     x                     x                    x
                #    x                     x                     x                    x
                #    _ooox                 oooox                xoooox               xooox
        ### Thậm chí máy còn có thể thua ngược nếu như không có quân x ở cuối chuỗi 3 
        if num3_Comp >= 2 and num3_Human == 0 and num3_Human_Block == 0:
            return 555555555555555555
        
        # Công thức tính điểm tổng quát dưới đây trong trường hợp 2 bên chưa chắc ai thắng có thể sẽ gặp nhiều sai sót, trong quá trình làm việc sẽ tiếp tục cập nhật
        # và đánh giá thay đổi hệ số của các chuỗi mỗi quân cờ
        total_Score_Comp = 2 * near_By_Comp + 20 * num2_Comp_Block + 100 * num2_Comp + 3000 * num3_Comp_Block + 300000 * num3_Comp + num4_Comp_Block * 30000000
        total_Score_Human = near_By_Human + 70 * num2_Human_Block + 250 * num2_Human + 4000 * num3_Human_Block  + num3_Human * 30000000
        return total_Score_Comp - total_Score_Human
    ## Trường hợp với người cũng tương tự
    else:
        if num4_Comp > 0 or num4_Comp_Block > 0:
            return 5555555555555555555
        if num4_Human > 0:
            return -5555555555555555555
        if num4_Human_Block >= 2:
            return -5555555555555555555
        if num4_Human_Block == 0 and num3_Comp > 0:
            return 555555555555555555
        if num3_Human >= 2 and num3_Comp == 0 and num3_Comp_Block == 0:
            return -555555555555555555
        total_Score_Human = 2 * near_By_Human + 70 * num2_Human_Block + 250 * num2_Human + 3000 * num3_Human_Block + 300000 * num3_Human + num4_Human_Block * 30000000
        total_Score_Comp = near_By_Comp + 20 * num2_Comp_Block + 100 * num2_Comp + 4000 * num3_Comp_Block + num3_Comp * 30000000
        return total_Score_Comp - total_Score_Human


# Hàm kiểm tra xem tọa độ có "tệ hay không"
def checkBad_Point(act, env_state):
    xx,yy = convert_to_2D(act)
    # 1
    if xx + 1 < NUMBER_ROWS and env_state[act+NUMBER_COLS] != 0:
        return False
    # 2
    if yy + 1 < NUMBER_COLS and env_state[act+1] != 0:
        return False
    # 3
    if xx > 0 and env_state[act-NUMBER_COLS] != 0:
        return False
    # 4
    if yy > 0 and env_state[act-1] != 0:
        return False
    # 5
    if xx + 1 < NUMBER_ROWS and yy + 1 < NUMBER_COLS and env_state[act+1+NUMBER_COLS] != 0:
        return False
    # 6
    if xx + 1 < NUMBER_ROWS and yy > 0 and env_state[act-1+NUMBER_COLS] != 0:
        return False
    # 7
    if yy + 1 < NUMBER_COLS and xx > 0 and env_state[act+1-NUMBER_COLS] != 0:
        return False
    # 8
    if yy > 0 and xx > 0 and env_state[act-1-NUMBER_COLS] != 0:
        return False
    return True
    
def minimax(env_state, depth, alpha, beta, player):
    # Lượt hiện tại
    turn = env_state[NUMBER_COLS*NUMBER_ROWS+2]

   
    if turn%2 == player:
        best = [-1, -math.inf]
    else:
        best = [-1, math.inf]
    # Nếu độ sâu giảm tới 0 ( đoán trước tối đa depth nước đi ) hoặc bàn cờ đã hết cờ thì trả về giá trị của bàn cờ
    if depth == 0 or check_ended(env_state) != -1:
        sc = evaluate(env_state, player)
        return [-1, sc]
   
    val_act = np.where(env_state[0:NUMBER_ACTIONS]==0)[0]
    for act in val_act:
       
        # Bỏ qua nếu tọa độ đưa vào đủ " tệ "
        if checkBad_Point(act,env_state):
            continue
        # Nhét nước đi này vào kho chứa các nước đã đi của player để có thể đánh giá bàn cờ ở hàm evaluate(state, player, x, y)
        env = next_step(act,env_state)
        
        score = minimax(env, depth - 1, alpha, beta, player)
        score[0]= act
        # Cập nhật alpha và beta sau mỗi lần tìm kiếm trong 1 nhánh của minimax
        if turn%2 == player:
            if score[1] > best[1]:
                best = score
            alpha = max(alpha, best[1])
        else:
            if score[1] < best[1]:
                best = score
            beta = min(beta, best[1])

        if beta <= alpha:
            break  # Cắt tỉa alpha - beta
    return best

def cpu_bot_greedy(p_state, per):

    turn = np.count_nonzero(p_state[0:NUMBER_ACTIONS]) 

    if(turn==0):
        return np.random.randint(0, NUMBER_ACTIONS),per
    env_state = np.full(ENV_SIZE,0)
    env_state[0:NUMBER_ACTIONS+2]=p_state[0:NUMBER_ACTIONS+2]
    env_state[NUMBER_COLS*NUMBER_ROWS+2]=turn
    player= turn%2
    depth = 4 

    move = minimax(env_state, depth, -math.inf, math.inf, player)
    act_idx=move[0]
    return act_idx, per



