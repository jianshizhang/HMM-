import numpy as np
import hmm_viterbi
import max_probability_seg

def cal_hmm_matrix(observation):
    # 得到所有标签
    word_pos_file = open('ChineseDic.txt').readlines()
    tags_num = {}
    for line in word_pos_file:
        word_tags = line.strip().split(',')[1:]
        for tag in word_tags:
            if tag not in tags_num.keys():
                tags_num[tag] = 0
    tags_list = list(tags_num.keys())

    # 转移矩阵、发射矩阵
    transaction_matrix = np.zeros((len(tags_list), len(tags_list)), dtype=float)
    emission_matrix = np.zeros((len(tags_list), len(observation)), dtype=float)

    # 计算转移矩阵和发射矩阵
    word_file = open('199801.txt').readlines()
    for line in word_file:
        if line.strip() != '':
            word_pos_list = line.strip().split('  ')
            for i in range(1, len(word_pos_list)):
                tag = word_pos_list[i].split('/')[1]
                pre_tag = word_pos_list[i - 1].split('/')[1]
                try:
                    transaction_matrix[tags_list.index(pre_tag)][tags_list.index(tag)] += 1
                    tags_num[tag] += 1
                except ValueError:
                    if ']' in tag:
                        tag = tag.split(']')[0]
                    else:
                        pre_tag = tag.split(']')[0]
                    transaction_matrix[tags_list.index(pre_tag)][tags_list.index(tag)] += 1
                    tags_num[tag] += 1

            for o in observation:
                # 注意这里用in去找（' 我/'，' **我/'的区别），用空格和‘/’才能把词拎出来
                if ' ' + o in line:
                    pos_tag = line.strip().split(o)[1].split('  ')[0].strip('/')
                    if ']' in pos_tag:
                        pos_tag = pos_tag.split(']')[0]
                    emission_matrix[tags_list.index(pos_tag)][observation.index(o)] += 1

    for row in range(transaction_matrix.shape[0]):
        n = np.sum(transaction_matrix[row])
        transaction_matrix[row] += 1e-16
        transaction_matrix[row] /= n + 1

    for row in range(emission_matrix.shape[0]):
        emission_matrix[row] += 1e-16
        emission_matrix[row] /= tags_num[tags_list[row]] + 1

    times_sum = sum(tags_num.values())
    for item in tags_num.keys():
        tags_num[item] = tags_num[item] / times_sum

    # 返回隐状态，初始概率，转移概率，发射矩阵概率
    return tags_list, list(tags_num.values()), transaction_matrix, emission_matrix


if __name__ == '__main__':
    input_str = "我是中国人。"
    obs = max_probability_seg.seg(input_str).strip().split(' ')
    print(obs,type(obs))

    hid, init_p, trans_p, emit_p = cal_hmm_matrix(obs)
    result = hmm_viterbi.viterbi(len(obs), len(hid), init_p, trans_p, emit_p)
    tag_line = ''
    for k in range(len(result)):
        tag_line += obs[k] + hid[int(result[k])] + ' '
    print(tag_line)