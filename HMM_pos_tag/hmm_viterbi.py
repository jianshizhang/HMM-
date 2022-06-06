import numpy as np
def viterbi(obs_len, states_len, init_p, trans_p, emit_p):
    """
    :param obs_len: 观测序列长度 int
    :param states_len: 隐含序列长度 int
    :param init_p:初始概率 list
    :param trans_p:转移概率矩阵 np.ndarray
    :param emit_p:发射概率矩阵 np.ndarray
    :return:最佳路径 np.ndarray
    """
    max_p = np.zeros((states_len, obs_len))  # max_p每一列为当前观测序列不同隐状态的最大概率
    path = np.zeros((states_len, obs_len))  # path每一行存储上max_p对应列的路径

    # 初始化max_p第1个观测节点不同隐状态的最大概率并初始化path从各个隐状态出发
    for i in range(states_len):
        max_p[i][0] = init_p[i] * emit_p[i][0]
        path[i][0] = i

    # 遍历第1项后的每一个观测序列，计算其不同隐状态的最大概率
    for obs_index in range(1, obs_len):
        new_path = np.zeros((states_len, obs_len))
        # 遍历其每一个隐状态
        for hid_index in range(states_len):
            # 根据公式计算累计概率，得到该隐状态的最大概率
            max_prob = -1
            pre_state_index = 0
            for i in range(states_len):
                each_prob = max_p[i][obs_index - 1] * trans_p[i][hid_index] * emit_p[hid_index][obs_index]

                if each_prob > max_prob:
                    max_prob = each_prob
                    pre_state_index = i

            # 记录最大概率及路径
            max_p[hid_index][obs_index] = max_prob
            for m in range(obs_index):
                # "继承"取到最大概率的隐状态之前的路径（从之前的path中取出某条路径）
                new_path[hid_index][m] = path[pre_state_index][m]
            new_path[hid_index][obs_index] = hid_index
        # 更新路径
        path = new_path

    # 返回最大概率的路径
    max_prob = -1
    last_state_index = 0
    for hid_index in range(states_len):
        if max_p[hid_index][obs_len - 1] > max_prob:
            max_prob = max_p[hid_index][obs_len - 1]
            last_state_index = hid_index
    return path[last_state_index]


if __name__ == '__main__':
    hidden_state = ['AT', 'BEZ', 'IN', 'NN', 'VB', 'PERIOD']  # 隐状态
    observation = ['The', 'bear', 'is', 'on', 'the', 'move', '.']  # 观测序列

    # 初始状态
    start_probability = [0.2, 0.1, 0.1, 0.2, 0.3, 0.1]
    # 转移概率
    transaction_probability = np.array(
        [[1 / 48659, 1 / 48659, 1 / 48659, 48636 / 48659, 1 / 48659, 19 / 48659],
         [1937 / 2590, 1 / 2590, 426 / 2590, 187 / 2590, 1 / 2590, 38 / 2590],
         [43322 / 62148, 1 / 62148, 1325 / 62148, 17314 / 62148, 1 / 62148, 185 / 62148],
         [1067 / 81036, 3720 / 81036, 42470 / 81036, 11773 / 81036, 614 / 81036, 21392 / 81036],
         [6082 / 14009, 42 / 14009, 4758 / 14009, 1476 / 14009, 129 / 14009, 1522 / 14009],
         [8016 / 15031, 75 / 15031, 4656 / 15031, 1329 / 15031, 954 / 15031, 1 / 15031]])
    # 发射概率
    emission_probability = np.array(
        [[69016 / 69023, 1 / 69023, 1 / 69023, 1 / 69023, 69016 / 69023, 1 / 69023, 1 / 69023],
         [1 / 10072, 1 / 10072, 10065 / 10072, 1 / 10072, 1 / 10072, 1 / 10072, 1 / 10072],
         [1 / 5491, 1 / 5491, 1 / 5491, 5484 / 5491, 1 / 5491, 1 / 5491, 1 / 5491],
         [1 / 543, 10 / 543, 1 / 543, 1 / 543, 1 / 543, 36 / 543, 1 / 543],
         [1 / 187, 43 / 187, 1 / 187, 1 / 187, 1 / 187, 133 / 187, 1 / 187],
         [1 / 15031, 1 / 15031, 1 / 15031, 1 / 15031, 1 / 15031, 1 / 15031, 48809 / 15031]])
    result = viterbi(len(observation), len(hidden_state),
                     start_probability, transaction_probability, emission_probability)

    tag_line = ''
    for k in range(len(result)):
        tag_line += observation[k] + "/" + hidden_state[int(result[k])] + ' '
    print(tag_line)