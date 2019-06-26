# -*- coding: utf-8 -*-
# 张嘉玮
# 20190503
# 遗传算法求多极小函数最优值

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 5*np.sin(6*x) + 6*np.cos(5*x)

class GeneticAlgorithm(object):
    """
    遗传算法求多极小函数最优值
    """
    def __init__(self, cross_rate, mutation_rate, n_population, n_iterations, DNA_size):
        self.cross_rate = cross_rate     #交配的可能性大小
        self.mutate_rate = mutation_rate #基因突变率
        self.n_population = n_population #种群大小
        self.n_iterations = n_iterations #迭代次数
        self.DNA_size = 6              # DNA的长度
        self.x_bounder = [-3, 6]         #搜索范围

    def init_population(self):
        """
        初始化种群
        :return:
        """
        population = np.random.randint(low=0, high=2, size=(self.n_population, self.DNA_size)).astype(np.int8)
        return population

    def transformDNA(self, population):
        """
        编码：十进制转化为二进制
        :param population:
        :return:
        """
        population_decimal = ( (population.dot(np.power(2, np.arange(self.DNA_size)[::-1])) / np.power(2, self.DNA_size) - 0.5) *
                               (self.x_bounder[1] - self.x_bounder[0]) + 0.5 * (self.x_bounder[0] + self.x_bounder[1])  )
        return population_decimal

    def fitness(self, population):
        """
        适应性函数
        :param population: 基因个体
        :return: 与当前最优的解
        """
        transform_population = self.transformDNA(population)
        fitness_score = f(transform_population)
        return fitness_score - fitness_score.min()

    def select(self, population, fitness_score):
        """
        根据适应值进行选择
        :param population:
        :param fitness_score:
        :return:
        """
        fitness_score = fitness_score + 1e-4

        fitness_score = np.array([1/i for i in fitness_score]) #求最小值，需要反一下，值越小，越适应
        idx = np.random.choice(np.arange(self.n_population), size=self.n_population, replace=True, p=fitness_score/fitness_score.sum())
        return population[idx]

    def create_child(self, parent, pop):
        """
        交叉
        :param parent:
        :param pop:
        :return:
        """
        if np.random.rand() < self.cross_rate:
            index = np.random.randint(0, self.n_population, size=1)
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)
            parent[cross_points] = pop[index, cross_points]
        return parent

    def mutate_child(self, child):
        """
        变异
        :param child:
        :return:
        """
        for i in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                child[i] = 1
            else:
                child[i] = 0
        return child

    # 进化
    def evolution(self):
        population = self.init_population()
        NUM = []
        for i in range(self.n_iterations):
            fitness_score = self.fitness(population)
            best_person = population[np.argmin(fitness_score)]
            if (i+1)%100 == 0:
                NUM.append(f(self.transformDNA(best_person)))
                print(u'第%-4d次进化后, 最优结果在: %s处取得, 对应的最小值为: %f' % (i, self.transformDNA(
                                                                                            best_person),
                                                                                        f(self.transformDNA(
                                                                                            best_person))))

            population = self.select(population, fitness_score)
            population_copy = population.copy()

            for parent in population:
                child = self.create_child(parent, population_copy)
                child = self.mutate_child(child)
                parent[:] = child

            population = population

        x_num = [i for i in range(len(NUM))]
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        plt.figure(1)
        plt.title('遗传算法')
        plt.ylabel('适应度（最小值）')
        plt.xlabel('进化次数（*100）')
        plt.plot(x_num, NUM, 'g')
        plt.show()

        self.best_person = best_person
        return self.transformDNA(best_person)

def main():
    ga = GeneticAlgorithm(cross_rate=0.9, mutation_rate=0.1, n_population=200, n_iterations=2001, DNA_size=8)
    best = ga.evolution()

    # 绘图
    plt.figure(2)
    x = np.linspace(start=ga.x_bounder[0], stop=ga.x_bounder[1], num=200)
    plt.plot(x, f(x))

    plt.scatter(ga.transformDNA(ga.best_person), f(ga.transformDNA(ga.best_person)), s=200, lw=0, c='red', alpha=0.5)
    ax = plt.gca()

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.show()
    return best

if __name__ == '__main__':
    Bests = []
    for _ in range(1):
        Bests.append(f(main()))
    Avg = np.mean(Bests)
    Best = np.min(Bests)
    Worst = np.max(Bests)
    Var = np.var(Bests)
    print("结果:", Bests)
    print("\nAVG: ", Avg,
          "\nBest: ", Best,
          "\nWorst: ", Worst,
          "\nVar: ", Var)

