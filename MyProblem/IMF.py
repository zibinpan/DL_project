# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea


class IMF1(ea.Problem):  # 继承Problem父类
    def __init__(self, Dim=30):
        name = 'IMF1'  # 初始化name（函数名称，可以随意设置）
        M = 2  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * Dim  # 决策变量下界
        ub = [1] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        t = (1 + 5 * np.tile(np.arange(2, self.Dim+1), (pop.sizes, 1))/self.Dim) * pop.Phen[:, 1:] - \
            np.tile(pop.Phen[:, 0: 1], (1, self.Dim-1))
        g = 1 + 9 * np.mean(t**2, axis=1, keepdims=True)
        pop.ObjV = np.c_[pop.Phen[:, 0:1],
                        g * (1 - np.sqrt(pop.Phen[:, 0:1] / g))]
    
    def calReferObjV(self):
        N = 10000  # 生成10000个参考点
        f1 = np.linspace(0, 1, N*self.M)
        f2 = 1 - np.sqrt(f1)
        referenceObjV = np.c_[f1, f2]
        return referenceObjV

class IMF2(ea.Problem):  # 继承Problem父类
    def __init__(self, Dim=30):
        name = 'IMF2'  # 初始化name（函数名称，可以随意设置）
        M = 2  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * Dim  # 决策变量下界
        ub = [1] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        t = (1 + 5 * np.tile(np.arange(2, self.Dim+1), (pop.sizes, 1))/self.Dim) * pop.Phen[:, 1:] - \
            np.tile(pop.Phen[:, 0: 1], (1, self.Dim-1))
        g = 1 + 9 * np.mean(t**2, axis=1, keepdims=True)
        pop.ObjV = np.c_[pop.Phen[:, 0:1],
                        g * (1 - (pop.Phen[:, 0:1] / g)**2)]
    
    def calReferObjV(self):
        N = 10000  # 生成10000个参考点
        f1 = np.linspace(0, 1, N*self.M)
        f2 = 1 - f1**2
        referenceObjV = np.c_[f1, f2]
        return referenceObjV

class IMF3(ea.Problem):  # 继承Problem父类
    def __init__(self, Dim=30):
        name = 'IMF3'  # 初始化name（函数名称，可以随意设置）
        M = 2  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * Dim  # 决策变量下界
        ub = [1] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        n, d = np.shape(pop.Phen)
        t = (1 + 5 * np.tile(np.arange(2, d+1), (n, 1))/d) * pop.Phen[:, 1:] - \
            np.tile(pop.Phen[:, 0: 1], (1, d-1))
        g = 1 + 9 * np.mean(t**2, axis=1, keepdims=True)
        temp = 1 - np.exp(-4*pop.Phen[:, 0: 1]) * np.sin(6*np.pi*pop.Phen[:, 0:1])**6
        pop.ObjV = np.c_[temp,
                        g * (1 - (temp / g)**2)]
    
    def calReferObjV(self):
        ref_num = 10000
        minf1 = np.amin(1 - np.exp(-4*np.linspace(0, 1, 1000000)) *
                        (np.sin(6 * np.pi * np.linspace(0, 1, 1000000)))**6)
        f1 = (np.linspace(minf1, 1, ref_num*self.M))
        f2 = 1 - f1**2
        return np.c_[f1, f2]