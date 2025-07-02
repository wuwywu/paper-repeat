# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2025/1/16
# User      : WuY
# File      : algorithm.py
# 文件中包含：


import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange


@njit
def Euler(fun, x0, t, dt, *args):
    """
    使用 euler 方法计算一个时间步后系统的状态。
    args:
        fun: 微分方程
        x0: 上一个时间单位的状态变量
        t: 运行时间
        dt: 时间步长
    :return: 
        x0 (numpy.ndarray): 下一个时间单位的状态变量
    """
    # 计算下一个时间单位的状态变量
    x0 += dt * fun(x0, t, *args)
    return x0

@njit
def Heun(fun, x0, t, dt, *args):
    """
    使用 Heun 方法计算一个时间步后系统的状态。
    args:
        fun: 微分方程函数，形式为 fun(x, t, *args)
        x0: 上一个时间单位的状态变量 (numpy.ndarray)
        t: 当前时间
        dt: 时间步长
    return:
        x1 (numpy.ndarray): 下一个时间单位的状态变量
    """
    # 计算当前点的斜率
    k1 = fun(x0, t, *args)
    
    # 使用 Euler 法预测值
    x_pred = x0 + dt * k1
    
    # 在预测点上计算新的斜率
    k2 = fun(x_pred, t + dt, *args)
    
    # 加权平均斜率得到新的状态
    x0 += 0.5 * dt * (k1 + k2)
    return x0

@njit
def RK4(fun, x0, t, dt, *args):
    """
    使用 Runge-Kutta 方法计算一个时间步后系统的状态。
    args:
        fun: 微分方程
        x0: 上一个时间单位的状态变量
        t: 运行时间
        dt: 时间步长
    :return:
        x0 (numpy.ndarray): 下一个时间单位的状态变量
    """
    k1 = fun(x0, t, *args)
    k2 = fun(x0 + (dt / 2.) * k1, t + (dt / 2.), *args)
    k3 = fun(x0 + (dt / 2.) * k2, t + (dt / 2.), *args)
    k4 = fun(x0 + dt * k3, t + dt, *args)

    x0 += (dt / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x0

@njit
def RKF45(fun, x0, t, dt, *args):
    """
    使用 Runge-Kutta-Fehlberg 45 方法计算一个时间步后的系统状态（固定步长）。
    
    输入：
        fun: 微分方程函数 fun(x, t, *args)
        x0 : 当前状态 (numpy.ndarray)
        t  : 当前时间
        dt : 时间步长
        args: 额外参数
    输出：
        x0 : 下一个时间步的状态（使用五阶公式）
    """
    # 每个子步骤的时间因子
    c2 = 1/4
    c3 = 3/8
    c4 = 12/13
    c5 = 1.0
    c6 = 1/2

    # 子步骤的系数（Butcher tableau）
    a21 = 1/4

    a31 = 3/32
    a32 = 9/32

    a41 = 1932/2197
    a42 = -7200/2197
    a43 = 7296/2197

    a51 = 439/216
    a52 = -8
    a53 = 3680/513
    a54 = -845/4104

    a61 = -8/27
    a62 = 2
    a63 = -3544/2565
    a64 = 1859/4104
    a65 = -11/40

    # 五阶系数（用于最终更新）
    b1 = 16/135
    b2 = 0
    b3 = 6656/12825
    b4 = 28561/56430
    b5 = -9/50
    b6 = 2/55

    k1 = dt * fun(x0, t, *args)
    k2 = dt * fun(x0 + a21 * k1, t + c2 * dt, *args)
    k3 = dt * fun(x0 + a31 * k1 + a32 * k2, t + c3 * dt, *args)
    k4 = dt * fun(x0 + a41 * k1 + a42 * k2 + a43 * k3, t + c4 * dt, *args)
    k5 = dt * fun(x0 + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4, t + c5 * dt, *args)
    k6 = dt * fun(x0 + a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5, t + c6 * dt, *args)

    x0 += b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6

    return x0

@njit
def discrete(fun, x0, t, dt, *args):
    """
    使用离散方法计算一个时间步后系统的状态。
    args:
        fun: 微分方程
        x0: 上一个时间单位的状态变量
        t: 运行时间
        dt: 时间步长(设定为1)
    :return:
        x0 (numpy.ndarray): 下一个时间单位的状态变量
    """
    x0[:] = fun(x0, t, *args)
    return x0
