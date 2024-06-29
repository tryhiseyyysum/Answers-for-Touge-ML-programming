import numpy as np
from scipy import stats


def em_single(init_values, observations):
    """
    模拟抛掷硬币实验并估计在一次迭代中，硬币A与硬币B正面朝上的概率
    :param init_values:硬币A与硬币B正面朝上的概率的初始值，类型为list，如[0.2, 0.7]代表硬币A正面朝上的概率为0.2，硬币B正面朝上的概率为0.7。
    :param observations:抛掷硬币的实验结果记录，类型为list。
    :return:将估计出来的硬币A和硬币B正面朝上的概率组成list返回。如[0.4, 0.6]表示你认为硬币A正面朝上的概率为0.4，硬币B正面朝上的概率为0.6。
    """

    #********* Begin *********#
    # 初始化
    p_A, p_B = init_values
    # E step
    heads_A, tails_A = 0, 0     # 硬币A正反面次数
    heads_B, tails_B = 0, 0     # 硬币B正反面次数   
    for observation in observations:        # observation代表一轮实验的结果
        len_observation = len(observation)  # 每轮实验抛掷的次数
        num_heads = observation.count(1)          # 正面次数
        num_tails = len_observation - num_heads     # 反面次数
        # 计算硬币A和硬币B产生当前观测数据的概率
        weight_A = stats.binom.pmf(num_heads, len_observation, p_A)  # 二项分布概率质量函数，返回硬币A产生当前观测数据的概率,num_heads为正面次数,len_observation为抛掷次数,p_A为硬币A正面朝上的概率
        weight_B = stats.binom.pmf(num_heads, len_observation, p_B)
        #归一化概率(每一轮都要归一化，因为每一轮的权重和不为1)
        norm_weight_A = weight_A / (weight_A + weight_B)
        norm_weight_B = weight_B / (weight_A + weight_B)
        # 计算硬币A和硬币B产生当前观测数据的概率之比
        heads_A += norm_weight_A * num_heads    #求期望
        tails_A += norm_weight_A * num_tails
        heads_B += norm_weight_B * num_heads
        tails_B += norm_weight_B * num_tails
    # M step
    p_A = heads_A / (heads_A + tails_A)     # 更新硬币A正面朝上的概率
    p_B = heads_B / (heads_B + tails_B)     # 更新硬币B正面朝上的概率
    return [p_A, p_B]
    #********* End *********#

#测试
init_values = [0.2, 0.7]
observations = [[1, 1, 0, 1, 0], [0, 0, 1, 1, 0], [1, 0, 0, 0, 0], [1, 0, 0, 1, 1], [0, 1, 1, 0, 0]]
print(em_single(init_values, observations))  #输出[0.34654779620869536, 0.5287058763827479]


# import numpy as np
# from scipy import stats


# def em_single(init_values, observations):
#     """
#     模拟抛掷硬币实验并估计在一次迭代中，硬币A与硬币B正面朝上的概率。请不要修改！！
#     :param init_values:硬币A与硬币B正面朝上的概率的初始值，类型为list，如[0.2, 0.7]代表硬币A正面朝上的概率为0.2，硬币B正面朝上的概率为0.7。
#     :param observations:抛掷硬币的实验结果记录，类型为list。
#     :return:将估计出来的硬币A和硬币B正面朝上的概率组成list返回。如[0.4, 0.6]表示你认为硬币A正面朝上的概率为0.4，硬币B正面朝上的概率为0.6。
#     """
#     observations = np.array(observations)
#     counts = {'A': {'H': 0, 'T': 0}, 'B': {'H': 0, 'T': 0}}
#     theta_A = init_values[0]
#     theta_B = init_values[1]
#     # E step
#     for observation in observations:
#         len_observation = len(observation)
#         num_heads = observation.sum()
#         num_tails = len_observation - num_heads
#         # 两个二项分布
#         contribution_A = stats.binom.pmf(num_heads, len_observation, theta_A)
#         contribution_B = stats.binom.pmf(num_heads, len_observation, theta_B)
#         weight_A = contribution_A / (contribution_A + contribution_B)
#         weight_B = contribution_B / (contribution_A + contribution_B)
#         # 更新在当前参数下A、B硬币产生的正反面次数
#         counts['A']['H'] += weight_A * num_heads
#         counts['A']['T'] += weight_A * num_tails
#         counts['B']['H'] += weight_B * num_heads
#         counts['B']['T'] += weight_B * num_tails
#     # M step
#     new_theta_A = counts['A']['H'] / (counts['A']['H'] + counts['A']['T'])
#     new_theta_B = counts['B']['H'] / (counts['B']['H'] + counts['B']['T'])
#     return [new_theta_A, new_theta_B]

#EM算法的迭代过程
def em(observations, thetas, tol=1e-4, iterations=100):
    """
    模拟抛掷硬币实验并使用EM算法估计硬币A与硬币B正面朝上的概率。
    :param observations: 抛掷硬币的实验结果记录，类型为list。
    :param thetas: 硬币A与硬币B正面朝上的概率的初始值，类型为list，如[0.2, 0.7]代表硬币A正面朝上的概率为0.2，硬币B正面朝上的概率为0.7。
    :param tol: 差异容忍度，即当EM算法估计出来的参数theta不怎么变化时，可以提前挑出循环。例如容忍度为1e-4，则表示若这次迭代的估计结果与上一次迭代的估计结果之间的L1距离小于1e-4则跳出循环。为了正确的评测，请不要修改该值。
    :param iterations: EM算法的最大迭代次数。为了正确的评测，请不要修改该值。
    :return: 将估计出来的硬币A和硬币B正面朝上的概率组成list或者ndarray返回。如[0.4, 0.6]表示你认为硬币A正面朝上的概率为0.4，硬币B正面朝上的概率为0.6。
    """

    #********* Begin *********#
    theta = thetas
    for i in range(iterations):
        new_theta = em_single(theta, observations)
        if sum(abs(np.array(new_theta) - np.array(theta))) < tol:
            break
        theta = new_theta
    return theta
    #********* End *********#