import random
import matplotlib.pyplot as plt
import math as m

# 定义目标函数
def target_function(x):
    return x[0] ** 2 + x[1] ** 2 + 8


# 定义约束条件
# x_1^2 - x_2 > 0
def constraint_function_1(x):
    return x[0] ** 2 - x[1]


# -x_1 - x_2^2 +2 == 0
def constraint_function_2(x):
    return -x[0] - x[1] ** 2 + 2


def decode(individual):
    x1 = decode_gene_to_real(individual.gene[0])
    x2 = decode_gene_to_real(individual.gene[1])
    return [x1, x2]


# 定义解码函数 - 将编码空间基因映射到解空间
def decode_gene_to_real(gene):
    # 解空间区间
    lower_bound = 0
    upper_bound = 9
    val = int(''.join(map(str, gene)), 2)
    max_val = 2 ** len(gene) - 1
    return lower_bound + (val / max_val) * (upper_bound - lower_bound)


# 适应度函数，包括约束的惩罚
def fitness(x, alpha=10, beta=1000):
    # 约束检查 Constraint Check
    c1 = constraint_function_1(x)
    c2 = constraint_function_2(x)
    if c1 <= 0 or not m.isclose(c2, 0, abs_tol=1e-3):
        c1_penalty = 0
        c2_penalty = 0
        if c1 <= 0:  # 弱约束
            c1_penalty = alpha * abs(c1)
        if not m.isclose(c2, 0, abs_tol=1e-3):  # 强约束
            c2_penalty = beta * abs(c2)

        # 最小值问题，fitness是以大为优，需要翻转
        fitness_after_penalty = target_function(x) + c1_penalty + c2_penalty
        return -fitness_after_penalty  # penalty 严重惩罚不满足约束的个体

    return -target_function(x)


# 定义个体，代表种群中的一个个体,个体的属性由gene,fitness组成
class Individual:
    def __init__(self, gene):
        self.gene = gene
        self.fitness = self.calc_fitness()

    # 计算适应度函数，这里以基因的平方和为例
    # 适应度函数应根据具体问题进行定义
    def calc_fitness(self):
        x = decode(self)
        return fitness(x)


# 种群初始化
def initialize_population():
    # POPULATION_SIZE: 种群的大小
    # GENES_LENGTH: 个体基因序列的长度
    # 生成初始种群，每个个体由随机生成的基因序列组成

    return [Individual([[random.randint(0, 1) for _ in range(GENES_LENGTH)] for _ in range(2)]) for _ in
            range(POPULATION_SIZE)]


# 3. 定义GA基本操作
# 选择
def selection(population, num_of_parents):
    # 选择种群中适应度高的前 num_of_parents 个个体
    # (reverse=True)排列population中的individual.fitness(lambda x: x.fitness)
    sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
    return sorted_population[:num_of_parents]


# 交叉
def crossover(parent1, parent2):
    # 单点交叉
    # parent1, parent2: 选择的两个父本个体
    # 随机选择交叉点，交换父本基因，生成两个子代
    if random.random() < CROSSOVER_RATE:
        cross_point = random.randint(1, len(parent1.gene[0]) - 1)
        child1_gene = [parent1.gene[0][cross_point:] + parent2.gene[0][:cross_point]
            , parent1.gene[1][cross_point:] + parent2.gene[1][:cross_point]]

        child2_gene = [parent2.gene[0][cross_point:] + parent1.gene[0][:cross_point]
            , parent2.gene[1][cross_point:] + parent1.gene[1][:cross_point]]
        return Individual(child1_gene), Individual(child2_gene)
    else:
        return parent1, parent2


# 突变
def mutation(individual):
    # 对个体的基因序列进行随机变异
    # individual: 要变异的个体
    # MUTATION_RATE: 变异概率
    for dim in range(2):
        for i in range(len(individual.gene)):
            if random.random() < MUTATION_RATE:
                # gene序列中随机点发生突变
                individual.gene[dim][i] = 1 if individual.gene[i] == 0 else 0
                # 突变后更新fitness
    individual.fitness = individual.calc_fitness()


# GA超参数设置,consist of population size, genes length, generation times, mutation rate
POPULATION_SIZE = 1000
GENES_LENGTH = 17
MAX_GENERATIONS = 1000
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.6

history = {'generation': [], 'fitness': []}


# 接口主函数
def genetic_algorithm():
    # 1. 初始化种族
    population = initialize_population()

    global_best_individual = max(population, key=lambda x: x.fitness)
    global_best_gen = 0

    # 逻辑主循环
    for gen in range(MAX_GENERATIONS):
        # 2. 选择
        parents = selection(population, len(population) // 2)  # 选择一半的个体
        offspring = []
        while len(offspring) < POPULATION_SIZE:
            # 3. 交叉
            parent1, parent2 = random.sample(population, 2)
            child1, child2 = crossover(parent1, parent2)
            # 4. 突变
            mutation(child1)
            mutation(child2)
            offspring.extend([child1, child2])
        population = offspring

        # 择出当代种群最优个体
        local_best_individual = max(offspring, key=lambda x: x.fitness)
        # 择出历代种群最优个体
        if local_best_individual.fitness > global_best_individual.fitness:
            global_best_individual = local_best_individual
            global_best_gen = gen + 1

        print(f"第{gen + 1}次进化，种群中个体最优适应度: {local_best_individual.fitness}")
        history['generation'].append(gen + 1)
        history['fitness'].append(local_best_individual.fitness)
    return global_best_individual, global_best_gen


# main
best, best_gen = genetic_algorithm()
best_x = decode(best)
print(f"最优个体基因: {best.gene},最优个体适应度: {best.fitness},最优个体产生于第{best_gen}代")
print(f"解码后对应的最优解x={best_x},f(x)={target_function(best_x)}")
print(f"约束情况1：{constraint_function_1(best_x)};约束情况2：{constraint_function_2(best_x)}")

# plt
plt.plot(history['generation'], history['fitness'])
plt.title('GA')
plt.xlabel('gen')
plt.ylabel('fitness')
plt.show()