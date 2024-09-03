import random
from matplotlib import pyplot as plt

class SA:
    def __init__(self, target_func, MAX_GENERATION=10, T_START=100, T_END=0.01, ANNEAL_RATE=0.99):
        self.MAX_GENERATION = MAX_GENERATION
        self.T_START = T_START
        self.T_END = T_END
        self.ANNEAL_RATE = ANNEAL_RATE
        self.target_func = target_func
        self.T_cur = T_START
        self.var1_global_best = 0
        self.var2_global_best = 0
        self.E_global_best = float("inf")
        self.history = {'Energy': [], 'Temp': []}

    def generate_new(self, x, y):
        while True:
            # 在现有解基础上产生随机扰动
            x_new = x + self.T_cur * (random() - random())
            y_new = y + self.T_cur * (random() - random())
            # 检查约束
            if abs(x_new) < 5 and abs(y_new) < 5:
                break
        return x_new, y_new

    def Metropolis(self, E, E_new):  # Metropolis接收准则
        if E_new <= E:
            return 1
        else:
            p = math.exp((E - E_new) / self.T_cur)
            if random() < p:
                return 1
            else:
                return 0

    def simulate_annealing(self):
        # init
        var1_new, var2_new = self.generate_new(2, 2)
        var1_current_best, var2_current_best = var1_new, var2_new
        self.var1_global_best, self.var2_global_best = var1_current_best, var2_current_best

        # 外循环，退火
        while self.T_cur > self.T_END:

            E_current_best = self.target_func(var1_current_best, var2_current_best)

            # 内循环
            for i in range(self.MAX_GENERATION):

                # 1. 在现有解基础上产生随机扰动 2. 检查约束
                var1_new, var2_new = self.generate_new(var1_new, var2_new)

                # 退火过程
                E_new = self.target_func(var1_new, var2_new)  # 目标函数

                # 接受准则
                if self.Metropolis(E_current_best, E_new):
                    E_current_best = E_new
                    var1_current_best, var2_current_best = var1_new, var2_new

                    if E_new <= self.E_global_best:
                        self.E_global_best = E_new
                        self.var1_global_best, self.var2_global_best = var1_new, var2_new

            # 当前温度下粒子退火迭代gen次后，开始降温
            print(
                f"当前温度:{self.T_cur}, 当次退火局部最低能量:{E_current_best}， 对应变量：:{self.var1_global_best}， {self.var2_global_best}")

            # 记录
            self.history['Energy'].append(E_current_best)
            self.history['Temp'].append(self.T_cur)
            # 降温
            self.T_cur = self.T_cur * self.ANNEAL_RATE

            # 退火结束，此时已经获得最优解


def func(x, y):  # 函数优化问题
    res = 4 * x ** 2 - 2.1 * x ** 4 + x ** 6 / 3 + x * y - 4 * y ** 2 + 4 * y ** 4
    return res


# 主函数
sa = SA(func, MAX_GENERATION=100000, T_START=100, T_END=0.01, ANNEAL_RATE=0.99)
sa.simulate_annealing()

plt.plot(sa.history['Temp'], sa.history['Energy'])
plt.title('SA')
plt.xlabel('Temp')
plt.ylabel('Energy')
plt.gca().invert_xaxis()
plt.show()

print(f"全局最优解:x1={sa.var1_global_best}, x2={sa.var2_global_best}, 最优解目标值:{sa.E_global_best}")
