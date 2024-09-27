import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置参数
r = 1  # 球半径
n = 20  # theta 的分割数
m = 40  # phi 的分割数

# 生成 theta 和 phi 的值
theta = np.linspace(0, np.pi / 2, n)  # 只绘制上半球
phi = np.linspace(0, 2 * np.pi, m)

# 生成网格
theta, phi = np.meshgrid(theta, phi)

# 计算球面坐标并添加细微纹理
perturbation = 0.05 * np.sin(5 * theta) * np.cos(5 * phi)  # 细微的扰动
radius = r + perturbation

x = radius * np.sin(theta) * np.cos(phi)
y = radius * np.sin(theta) * np.sin(phi)
z = radius * np.cos(theta)

# 创建三维图形
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
ax.plot_surface(x, y, z, rstride=5, cstride=5, color='b', edgecolor='k', alpha=0.6)

# 设置坐标轴范围
max_range = np.max([np.max(x), np.max(y), np.max(z)])  # 计算最大范围
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([-max_range, max_range])

# 设置轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Upper Half of Eyeball Surface')

# 显示图形
plt.show()
