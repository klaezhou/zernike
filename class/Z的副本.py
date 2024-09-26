import numpy as np
from scipy.optimize import least_squares
import math

def modes(order):
    #生成n，m的列表
    list = np.empty((0, 2), int)
    for i in range(order+1) :
        for j in range(-i,i+1):
            if (i-abs(j)) % 2 == 0:
                list = np.vstack([list, [i, j]])
    return list

def zernike_radial(n, m, rho):
    #R_n^m(rho)
    if (n - m) % 2 != 0:
        return np.zeros_like(rho)
    sum = np.zeros_like(rho)
    for k in range((n - abs(m) )// 2 + 1):
        #print("m=", m, "n=", n,"k=",k)
        sum += ((-1)**k * math.factorial(n - k) /(math.factorial(k) * math.factorial((n + abs(m)) // 2 - k) * math.factorial((n - abs(m)) // 2 - k))) * rho**(n - 2 * k)
    return sum
def zernike_value(n, m, X, Y):
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)

    radial = zernike_radial(m, n, R)
    if m >= 0:
        return radial * np.cos(m * Theta)
    else:
        return radial * np.sin(-m * Theta)

def Z(data):
    mode = modes(6) #6阶zernike多项式
    n_data=len(data)
    n_polynomial=len(mode)
    Z=np.zeros(( n_polynomial,n_data))

    '''
    for i in range(n_data):
        for j in range(n_polynomial):
            Z[i,j]= zernike(0,0,data[i][0],data[i][1])
    '''
    x = [point[0] for point in data]
    y = [point[1] for point in data]

    Z_matrix = np.zeros((n_polynomial,n_data))

    # 计算每个 Zernike 模式在每个点 (x, y) 处的值

    for j in range(n_data):
        for i, (n, m) in enumerate(mode):
            Z_values = zernike_value(n, m, x[j], y[j])
            Z_matrix[i,j] = Z_values  # 将每个 Zernike 模式展平后存入 Z 矩阵的一列

    # 输出 Z 矩阵的形状和内容
    print("Z 矩阵形状:", Z_matrix.shape)

    return Z_matrix


data=[(2,3),(1,5),(3,5),(2,4)]
Z=Z(data)
print(Z)