import numpy as np

class utils:
    @staticmethod
    def generate_start_target(map_array, seed=40):
        np.random.seed(seed)
        while True:
            # 获取矩阵的形状
            num_rows, num_cols = map_array.shape

            # 随机选择起点和终点的行索引和列索引
            start_row, start_col = np.random.randint(0, num_rows), np.random.randint(0, num_cols)
            end_row, end_col = np.random.randint(0, num_rows), np.random.randint(0, num_cols)

            # 获取起点和终点的值
            start_point = map_array[start_row, start_col]
            end_point = map_array[end_row, end_col]


            if (start_row, start_col) == (end_row, end_col):
                continue

            if start_point == 0 and end_point == 0:
                break
        
        return (start_row, start_col) , (end_row, end_col)
    