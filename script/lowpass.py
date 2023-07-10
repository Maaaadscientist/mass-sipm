import numpy as np
import matplotlib.pyplot as plt

def lowpass_filter_response(w, wc, n):
    return 1 / np.sqrt(1 + (w / wc)**(2 * n))

# 设置截止频率和阶数的范围
cutoff_frequencies = [100, 200, 300]  # 截止频率
orders = [ 3, 4, 5]  # 阶数

# 生成频率范围
w = np.linspace(0, 1000, 10000)  # 频率范围

# 绘制频谱响应曲线图
for wc in cutoff_frequencies:
    for n in orders:
        response = lowpass_filter_response(w, wc, n)
        label = "Cutoff: {}Hz, Order: {}".format(wc, n)
        plt.plot(w, response, label=label)

# 设置图形参数
plt.xlabel('Frequency')
plt.ylabel('Response')
plt.title('Lowpass Filter Frequency Response')
plt.legend()
plt.grid(True)

# 显示图形
plt.show()

