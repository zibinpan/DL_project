# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea  # 导入geatpy库
from sys import path as paths
from os import path as path

paths.append(path.split(path.split(path.realpath(__file__))[0])[0])

from GNN_Model import generate_data

class moea_GNN_MOEA_templet(ea.MoeaAlgorithm):
    """
moea_NSGA3_DE_templet : class - 多目标进化优化GNN-MOEA算法
    
算法描述:
    采用GNN-MOEA进行多目标优化，
    """

    def __init__(self, problem, population):
        ea.MoeaAlgorithm.__init__(self, problem, population)  # 先调用父类构造方法
        if population.ChromNum != 1:
            raise RuntimeError('传入的种群对象必须是单染色体的种群类型。')
        self.name = 'GNN-MOEA'
        if self.problem.M < 10:
            self.ndSort = ea.ndsortESS  # 采用ENS_SS进行非支配排序
        else:
            self.ndSort = ea.ndsortTNS  # 高维目标采用T_ENS进行非支配排序，速度一般会比ENS_SS要快
        self.selFunc = 'tour'  # 基向量选择方式，采用锦标赛选择
        if population.Encoding == 'RI':
            self.mutOper = ea.Mutde(F=0.5)  # 生成差分变异算子对象
            self.recOper = ea.Xovbd(XOVR=0.5, Half_N=True)  # 生成二项式分布交叉算子对象，这里的XOVR即为DE中的Cr
        else:
            raise RuntimeError('编码方式必须为''RI''.')
        self.F = 0.5  # 差分变异缩放因子（可以设置为一个数也可以设置为一个列数与种群规模数目相等的列向量）
        self.pc = 0.2  # 交叉概率
        self.MAXSIZE = 10 * population.sizes

    def reinsertion(self, population, offspring, NUM, uniformPoint, globalNDSet):

        """
        描述:
            重插入个体产生新一代种群（采用父子合并选择的策略）。
            NUM为所需要保留到下一代的个体数目。
            
        """

        # 父子两代合并
        population = population + offspring
        globalNDSet = population + globalNDSet
        # 选择个体保留到下一代
        [levels, criLevel] = self.ndSort(population.ObjV, NUM, None, population.CV,
                                         self.problem.maxormins)  # 对NUM个个体进行非支配分层
        chooseFlag = ea.refselect(population.ObjV, levels, criLevel, NUM, uniformPoint,
                                  self.problem.maxormins)  # 根据参考点的“入龛”个体筛选
        # 更新全局存档
        [levels, criLevel] = self.ndSort(globalNDSet.ObjV, None, None, globalNDSet.CV, self.problem.maxormins)
        globalNDSet = globalNDSet[np.where(levels == 1)[0]]
        if globalNDSet.CV is not None: # CV不为None说明有设置约束条件
            globalNDSet = globalNDSet[np.where(np.all(globalNDSet.CV <= 0, 1))[0]] # 排除非可行解
        if globalNDSet.sizes > self.MAXSIZE:
            dis = ea.crowdis(globalNDSet.ObjV, np.ones(globalNDSet.sizes)) # 计算拥挤距离
            globalNDSet = globalNDSet[np.argsort(-dis)[:self.MAXSIZE]] # 根据拥挤距离选择符合个数限制的解保留在存档中
        return population[chooseFlag], globalNDSet

    def gnn_generate_off(self, globalNDSet, NUM):
        Chrom = generate_data(globalNDSet.Chrom, globalNDSet.Field[0, :], globalNDSet.Field[1, :], NUM)
        offspring = ea.Population(globalNDSet.Encoding, globalNDSet.Field, Chrom.shape[0], Chrom)
        offspring.Chrom = ea.mutpolyn(offspring.Encoding, offspring.Chrom, offspring.Field, DisI=100)
        return offspring

    def run(self, prophetPop=None):  # prophetPop为先知种群（即包含先验知识的种群）
        # ==========================初始化配置===========================
        population = self.population
        self.initialization()  # 初始化算法模板的一些动态参数
        # ===========================准备进化============================
        uniformPoint, NIND = ea.crtup(self.problem.M, population.sizes)  # 生成在单位目标维度上均匀分布的参考点集
        population.initChrom(NIND)  # 初始化种群染色体矩阵，此时种群规模将调整为uniformPoint点集的大小，initChrom函数会把种群规模给重置
        # 插入先验知识（注意：这里不会对先知种群prophetPop的合法性进行检查）
        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]  # 插入先知种群
        self.call_aimFunc(population)  # 计算种群的目标函数值
        [levels, criLevel] = self.ndSort(population.ObjV, NIND, None, population.CV,
                                         self.problem.maxormins)  # 对NIND个个体进行非支配分层
        population.FitnV = (1 / levels).reshape(-1, 1)  # 直接根据levels来计算初代个体的适应度
        globalNDSet = population[np.where(levels == 1)[0]]  # 创建全局存档，该全局存档贯穿进化始终，随着进化不断更新
        if globalNDSet.CV is not None:  # CV不为None说明有设置约束条件
            globalNDSet = globalNDSet[np.where(np.all(globalNDSet.CV <= 0, 1))[0]]  # 排除非可行解
        # ===========================开始进化============================
        while self.terminated(population) == False:
            # 进行差分进化操作
            offspring = population.copy()  # 存储子代种群
            offspring.Chrom = self.mutOper.do(offspring.Encoding, offspring.Chrom, offspring.Field)  # 变异
            tempPop = population + offspring  # 当代种群个体与变异个体进行合并（为的是后面用于重组）
            offspring.Chrom = self.recOper.do(tempPop.Chrom)  # 重组
            self.call_aimFunc(offspring)  # 计算目标函数值
            # if globalNDSet.sizes > 2 * NIND and self.currentGen > self.MAXGEN // 2 and self.currentGen % 10 == 0:
            #     offspring_gnn = self.gnn_generate_off(globalNDSet, NIND)
            #     offspring_gnn.Phen = offspring_gnn.Chrom.copy()
            #     self.problem.aimFunc(offspring_gnn)
            #     ea.moeaplot(offspring.ObjV)
            #     ea.moeaplot(offspring_gnn.ObjV)
            #     offspring += offspring_gnn
            # self.call_aimFunc(offspring)  # 计算目标函数值
            # 重插入生成新一代种群,同时更新全局存档
            population, globalNDSet = self.reinsertion(population, offspring, NIND, uniformPoint, globalNDSet)
        return self.finishing(population, globalNDSet)  # 调用finishing完成后续工作并返回结果
