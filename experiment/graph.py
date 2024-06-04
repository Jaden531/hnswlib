# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import sys

# 测试环境下的参数
max_elements = sys.argv[1]

data_dir = "../../gist_hnsw"

# 读取 x 轴数据和 y 轴数据
with open(data_dir + f"/qps/hnsw{max_elements}.txt", "r") as f:
    x_data = [float(line.strip()) for line in f.readlines()]

with open(data_dir + f"/recall/hnsw{max_elements}.txt", "r") as f:
    y_data = [float(line.strip()) for line in f.readlines()]

# 绘制折线图
plt.figure(figsize=(12, 8))
plt.plot(x_data, y_data, label='HNSW')

# 添加标题和标签
plt.title("Line Plot")
plt.xlabel("qps")
plt.ylabel("Recall")
plt.legend()

# 设置 x 轴刻度，使其粒度更精细
plt.xticks(range(int(min(x_data)), int(max(x_data))+1, 50),rotation=45)
plt.yticks([i/50.0 for i in range(int(min([int(y*50) for y in y_data])), 51)])
# 保存图形为图片文件
plt.savefig(data_dir + f"/graph/hnsw{max_elements}.png")

