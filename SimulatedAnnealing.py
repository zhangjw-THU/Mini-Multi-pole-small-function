# -*- coding: utf-8 -*-
# 张嘉玮
# 20190503
# 模拟退火算法求多极小函数

import matplotlib.pyplot as plt
import numpy as np

import math
import time

np.random.seed(int(time.time()))

num = 300
K = 0.1
R = 0.9     #控制温度降低快慢
T_max = 30  #初始温度
T_min = 0.1 #下限温度

def Func(x):
    return 5*np.sin(6*x) + 6*np.cos(5*x)


def main():

    x = np.random.uniform(0,2*math.pi)
    Best_A =Func(x)
    Best_array = []
    T_array = []
    T = T_max
    while(T >T_min):
        for i in range(num):
            x_temp = np.random.uniform(0,2*math.pi)
            current = Func(x_temp)
            dE = Best_A - current

            if dE>=0:
                Best_A = current
                x = x_temp
            else:
                if math.exp(dE/T) > np.random.uniform(0,1):
                    Best_A = current
                    x = x_temp

        T = R*T
        T_array.append(T)
        Best_array.append(Best_A)

    Plot(Best_array,T_array)
    # print("最小值 ： ",Best_A)
    return Best_A,x

def Plot(num,T_array):
    plt.figure(1)


    x_num = [i for i in range(len(num))]
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.subplot(211)
    plt.title('模拟退火算法')
    plt.ylabel('目标函数')
    plt.plot(x_num,num,'g')

    plt.subplot(212)
    plt.ylabel('温度')
    plt.plot(x_num,T_array,'r')

    plt.show()

if __name__ == '__main__':
    N = 1
    Bests = []
    xs = []
    for _ in range(N):
        y,x = main()
        Bests.append(y)
        xs.append(x)
    Avg = np.mean(Bests)
    Best = np.min(Bests)
    Worst = np.max(Bests)
    Var = np.var(Bests)
    print("结果:",Bests,"坐标：",xs)
    print("\nAVG: ",Avg,
          "\nBest: ",Best,
          "\nWorst: ",Worst,
          "\nVar: ",Var)

    plt.figure(2)
    x = np.linspace(start=0, stop=2*math.pi, num=200)
    plt.plot(x, Func(x))

    plt.scatter(xs, Bests, s=200, lw=0, c='red', alpha=0.5)
    ax = plt.gca()

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.show()

