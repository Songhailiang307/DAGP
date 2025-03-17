# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# 加载 PLINK 的 .raw 文件，分块读取以优化内存
def load_raw_data(file_path, chunksize=100000):
    reader = pd.read_csv(file_path, delim_whitespace=True, chunksize=chunksize)
    genotype_data = []
    for chunk in reader:
        genotype_data.append(chunk.iloc[:, 6:].values)  # 跳过前6列
    return np.vstack(genotype_data)

# One-Hot 编码函数，向量化实现
def one_hot_encode(genotype_data):
    # 展平数据
    flattened = genotype_data.flatten()
    # 初始化空矩阵用于存储 One-Hot 编码
    encoded_data = np.zeros((len(flattened), 3), dtype=int)
    # 对有效值进行编码
    valid_indices = flattened >= 0  # 忽略缺失值或错误值
    encoded_data[valid_indices, flattened[valid_indices].astype(int)] = 1
    # 重塑为原始数据形状
    return encoded_data.reshape(genotype_data.shape[0], -1)

# 拆分数据，将剩余部分加入最后一个块
def split_data(data, split_num):
    chunk_size = data.shape[1] // split_num
    remainder = data.shape[1] % split_num
    split_data = [data[:, i * chunk_size:(i + 1) * chunk_size] for i in range(split_num)]
    if remainder > 0:
        split_data[-1] = np.hstack([split_data[-1], data[:, -remainder:]])
    return split_data

# 主函数
def main(input_file_path, split_number):
    # 加载基因型数据
    print("Loading raw genotype data...")
    genotype_data = load_raw_data(input_file_path)
    print('Shape of data before One-Hot encoding:', genotype_data.shape)

    # One-Hot 编码并行化处理
    print("Encoding genotype data with One-Hot...")
    encoded_data = Parallel(n_jobs=4)(delayed(one_hot_encode)(chunk)
                                       for chunk in np.array_split(genotype_data, 4))
    encoded_genotype_data = np.vstack(encoded_data)
    print('Shape of data after One-Hot encoding:', encoded_genotype_data.shape)

    # 数据拆分
    print("Splitting data...")
    split_data_list = split_data(encoded_genotype_data, split_number)

    # 创建输出目录
    output_dir = 'SplitGenotypeData_OneHot'
    os.makedirs(output_dir, exist_ok=True)

    # 保存拆分后的数据
    for i, split_data_chunk in enumerate(split_data_list):
        print(f"Split {i}: Shape {split_data_chunk.shape}")
        output_file = os.path.join(output_dir, f"Genotype_Split_{i}.csv")
        np.savetxt(output_file, split_data_chunk, delimiter=",", fmt='%1.0f')

    print(f"Data successfully split into {split_number} parts and saved to '{output_dir}'.")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_file.raw> <split_number>")
        sys.exit(1)

    input_file_path = sys.argv[1]
    split_number = int(sys.argv[2])
    main(input_file_path, split_number)