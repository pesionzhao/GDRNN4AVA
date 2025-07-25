"""
@File :BasicForward.py
@Author :Pesion
@Date :2023/9/12
@Desc : 基于Zoeppritz方程的正演
"""

from abc import ABCMeta, abstractmethod
import numpy as np
from numpy import sin, cos, arcsin, arccos
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import cmath as cm


class ForwardModel(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, vp, vs, rho, theta, wavemat):  # TODO 未来可加入采样点个数，用于剪切vpvsrho
        """
        通过不同的正演方法获取反射系数

        Args:
            vp: 横波序列
            vs: 纵波序列
            rho: 密度
            theta: 入射角度集
            wavemat: 子波矩阵

        Returns:
            反射系数 rpp

        """
        pass

    @abstractmethod
    def showresult(self, dt0, trace, theta):
        """
        显示正演后的地震数据

        Args:
            dt0: 采样间隔
            trace: 展示地震数据的道数，单道默认为0
            theta: 横坐标角度制的theta


        """
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def jacobian(self, vp=None, vs=None, rho=None, theta=None, wav_mtx=None):
        pass
