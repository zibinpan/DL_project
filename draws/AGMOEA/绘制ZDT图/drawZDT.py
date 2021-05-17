# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import geatpy as ea # import geatpy
import numpy as np
from sys import path as paths
import time
import os

project_dir = 'D:/用户/桌面/homework/第二学期/6032/DL MOEA/code/Main/'
figures_dir = 'D:/用户/桌面/homework/第二学期/6032/DL MOEA/paper/figures/'  # figures存放的目录
paths.append(project_dir + 'Algorithms')
paths.append('D:/用户/桌面/homework/第二学期/6032/DL MOEA/code/GMOEA_refer/')

from moea_NSGEA2_archive_templet import moea_NSGEA2_archive_templet
from GMOEA import moea_GMOEA_templet

if __name__ == '__main__':
    names = ['ZDT1','ZDT2','ZDT3','ZDT4','ZDT6']
    NINDs = [6, 6, 6, 6, 6]
    MAXGENs = [300, 300, 300, 300, 300]
    exp_i = 3
    """================================实例化问题对象==========================="""
    problemName = names[exp_i]      # 问题名称
    fileName = problemName    # 这里因为目标函数写在与之同名的文件里，所以文件名也是问题名称
    MyProblem = getattr(__import__(fileName), problemName) # 获得自定义问题类
    problem = MyProblem()     # 生成问题对象
    """==================================种群设置=============================="""
    Encoding = 'RI'           # 编码方式
    NIND = NINDs[exp_i]                 # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND) # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """================================算法参数设置============================"""
    myAlgorithm = moea_NSGEA2_archive_templet(problem, population.copy()) # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = MAXGENs[exp_i] // 2  # 最大进化代数
    myAlgorithm.logTras = myAlgorithm.MAXGEN//10
    myAlgorithm.drawing = 0 # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    myAlgorithm1 = ea.moea_NSGA2_templet(problem, population.copy()) # 实例化一个算法模板对象
    myAlgorithm1.MAXGEN = MAXGENs[exp_i]  # 最大进化代数
    myAlgorithm1.logTras = myAlgorithm1.MAXGEN//10
    myAlgorithm1.drawing = 0 # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    myAlgorithm2 = ea.moea_MOEAD_templet(problem, population.copy()) # 实例化一个算法模板对象
    myAlgorithm2.MAXGEN = MAXGENs[exp_i]  # 最大进化代数
    myAlgorithm2.logTras = myAlgorithm2.MAXGEN//10
    myAlgorithm2.drawing = 0 # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    
    myAlgorithm3 = moea_GMOEA_templet(problem, population.copy()) # 实例化一个算法模板对象
    myAlgorithm3.MAXGEN = MAXGENs[exp_i]  # 最大进化代数
    myAlgorithm3.logTras = myAlgorithm3.MAXGEN//10
    myAlgorithm3.drawing = 0
    
    """===========================调用算法模板进行种群进化========================="""
    population.initChrom()  # 统一用相同的初始种群
    NDSet, _ = myAlgorithm.run(population.copy()) # 执行算法模板，得到非支配种群
    NDSet1, _ = myAlgorithm1.run(population.copy()) # 执行算法模板，得到非支配种群
    NDSet2, _ = myAlgorithm2.run(population.copy()) # 执行算法模板，得到非支配种群
    NDSet3, _ = myAlgorithm3.run(population.copy())
    # 计算指标
    PF = problem.getReferObjV() # 获取真实前沿，详见Problem.py中关于Problem类的定义
    plt.figure()
    ax = plt
    ax.scatter(PF[:, 0], PF[:, 1], alpha = 0.5, c = 'gray', s = 20, marker = '.', label = 'True Pareto Front')
    ax.scatter(NDSet.ObjV[:, 0], NDSet.ObjV[:, 1], c = 'orangered', s = 40, marker = 'o', label = 'AG-MOEA, IGD=' + str(format(myAlgorithm.log['igd'][-1], '.3f')) + ', HV = ' + str(format(myAlgorithm.log['hv'][-1], '.3f')))
    ax.scatter(NDSet1.ObjV[:, 0], NDSet1.ObjV[:, 1], c = 'green', s = 40, marker = 'x', label = 'NSGA-II, IGD=' + str(format(myAlgorithm1.log['igd'][-1], '.3f')) + ', HV = ' + str(format(myAlgorithm1.log['hv'][-1], '.3f')))
    ax.scatter(NDSet2.ObjV[:, 0], NDSet2.ObjV[:, 1], c = 'blue', s = 40, marker = '*', label = 'MOEA/D, IGD=' + str(format(myAlgorithm2.log['igd'][-1], '.3f')) + ', HV = ' + str(format(myAlgorithm2.log['hv'][-1], '.3f')))
    ax.scatter(NDSet3.ObjV[:, 0], NDSet3.ObjV[:, 1], c = 'yellow', s = 40, marker = '*', label = 'GMOEA, IGD=' + str(format(myAlgorithm3.log['igd'][-1], '.3f')) + ', HV = ' + str(format(myAlgorithm3.log['hv'][-1], '.3f')))
    ax.title(problemName+'\nN=' + str(NIND) + ', EVALS = ' + str(2 * NIND * myAlgorithm.MAXGEN))
    ax.xlabel('f1')
    ax.ylabel('f2')
    # ax.grid(True)
    ax.legend()
    plt.savefig(figures_dir+'Pareto_Front_of_' + problemName + '.pdf', dpi=100)
    plt.show()
    
    # 绘制IGD变化图
    metricName = ['igd']
    gens = np.array(myAlgorithm.log['gen']) * 2 * NIND
    Metric = np.array([myAlgorithm.log[metricName[i]] for i in range(len(metricName))])[0]
    gens1 = np.array(myAlgorithm1.log['gen']) * NIND
    Metric1 = np.array([myAlgorithm1.log[metricName[i]] for i in range(len(metricName))])[0]
    gens2 = np.array(myAlgorithm2.log['gen']) * NIND
    Metric2 = np.array([myAlgorithm2.log[metricName[i]] for i in range(len(metricName))])[0]
    gens3 = np.array(myAlgorithm3.log['gen']) * NIND
    Metric3 = np.array([myAlgorithm3.log[metricName[i]] for i in range(len(metricName))])[0]
    plt.figure()
    plt.plot(gens, Metric, 'o-', c='orangered', label='AG-MOEA')
    plt.plot(gens1, Metric1, '^-', c='green', label='NSGA-II')
    plt.plot(gens2, Metric2, '*-', c='blue', label='MOEA/D')
    plt.plot(gens3, Metric3, '*-', c='yellow', label='GMOEA')
    plt.title(problemName)
    plt.xlabel('Evaluation Number')
    plt.ylabel('IGD Value')
    plt.legend()
    plt.savefig(figures_dir + 'IGDs_' + problemName + '.pdf', dpi=100)
    plt.show()
    