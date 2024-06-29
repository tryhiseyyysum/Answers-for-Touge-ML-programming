from scipy.stats import binom

# 每个模型正确率
p1 = 0.51
p2 = 0.60
# 总模型数
n = 500
# 至少需要正确的次数
k = 251

# 计算 P(X >= 251)
probability_51 = 1 - binom.cdf(k - 1, n, p1) # 1 - P(X < 251),
probability_60 = 1 - binom.cdf(k - 1, n, p2)

print(probability_51, probability_60)
