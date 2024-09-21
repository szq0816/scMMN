import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
data = [[0.511,0.5116,0.5121,0.5119],
        [0.9933,0.9854,0.9085,0.9256],
        [0.501,0.6773,0.6786,0.5008],
        [0.853,0.8565,0.8336,0.8443],
        [0.8597,0.9121,0.475,0.8551],
        [0.707,0.4789,0.5254,0.425],
        [0.5009,0.5001,0.5098,0.6316],
        [0.9964,0.9956,0.9908,0.9817],
        [0.9956,0.996,0.9845,0.9897],
        [0.99,0.997,0.9872,0.9831],
        [0.9966,0.9981,0.9928,0.99]]
# x轴上坐标的个数，也是data列表里面的列表内的元素个数。4也可以用len(data[0])代替。
X = np.arange(4)
X[0] = 0
X[1] = 4
X[2] = 8
X[3] = 12

fig = plt.figure(figsize=(16, 8))
ax = fig.add_axes([0.1,0.1,0.8,0.8])

colors = plt.cm.viridis(np.linspace(0, 1, 11))

'''
F1、F2、F3画柱状图，共三个，第一个参数可以认为是labels，第二个参数是各自的y坐标，如果我们想要
在某个柱状图下面设置x轴刻度值，就可以在该柱状图下面用tick_label设置。
'''
F1 = ax.bar(X + 0.00, data[0], color = colors[0], width = 0.25)
F2 = ax.bar(X + 0.3, data[1], color = colors[1], width = 0.25)
F3 = ax.bar(X + 0.6, data[2], color = colors[2], width = 0.25)
F4 = ax.bar(X + 0.9, data[3], color = colors[3], width = 0.25)
F5 = ax.bar(X + 1.2, data[4], color = colors[4], width = 0.25)
F6 = ax.bar(X + 1.5, data[5], color = colors[5], width = 0.25,tick_label=["droput= -1.0","droput= -0.5","droput= 0.0","droput= 1.0"])
F7 = ax.bar(X + 1.8, data[6], color = colors[6], width = 0.25)
F8 = ax.bar(X + 2.1, data[7], color = colors[7], width = 0.25)
F9 = ax.bar(X + 2.4, data[8], color = colors[8], width = 0.25)
F10 = ax.bar(X + 2.7, data[9], color = colors[9], width = 0.25)
F11 = ax.bar(X + 3.0, data[10], color = colors[10], width = 0.25)

plt.ylim(0, 1)

# 给柱状图设置图例，第一个参数是个元组，元组中每个元素代表一个柱状图，第二个参数是对应着的图例的名字。
ax.legend((F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11),('$k$-means','GraphSCC','sc-INDC','scMCKC','AdClust','scCDG','DeepScena','scMAE','CAKE','Louvain','scMMN'),bbox_to_anchor=(1.01,1.1),ncol=11)

plt.savefig('simulate_FMI.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show()
