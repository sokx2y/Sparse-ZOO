import numpy as np

# 加载 .npz 文件
data = np.load('/capsule/home/xiangyuxing/Sparse-ZOO/medium_models/diff_linear_plots/layer_1_outputs.npz', allow_pickle=True)

# 获取文件中的数组名称
files = data.files

# 输出信息到文件
with open('output.txt', 'w') as f:
    # 将数组名称写入文件
    f.write("Arrays in the .npz file:\n")
    for file in files:
        f.write(f"{file}:")
        f.write(f"{data[file]}\n")

print("信息已写入 'output.txt'")

