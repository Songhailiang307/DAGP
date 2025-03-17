# 加载必要的库
require(BGLR)   # 用于贝叶斯回归和基因组选择


Args <- commandArgs(trailingOnly=TRUE)
genofile=Args[1]



# 读取基因型数据
genotype_data <- read.csv(genofile, header = FALSE)

# 检查基因型数据的前几行
head(genotype_data)

# 将基因型数据转换为矩阵（假设数据为二进制格式，1表示等位基因存在，0表示不存在）
genotype_matrix <- as.matrix(genotype_data)

# 检查基因型矩阵的大小
cat("Genotype matrix dimensions:", dim(genotype_matrix), "\n")

# 去除标准差为零的列
genotype_matrix <- genotype_matrix[, apply(genotype_matrix, 2, sd) > 0]

# 检查处理后的基因型矩阵维度
cat("Filtered genotype matrix dimensions:", dim(genotype_matrix), "\n")

# 计算基因组关系矩阵 (GRM)
# GRM = (X'X) / n
# 其中 X 是标准化后的基因型矩阵，n 是标记数（矩阵的列数）

grm <- tcrossprod(scale(genotype_matrix)) / ncol(genotype_matrix)


# 显示 GRM 的前几行
cat("Subset of Genomic Relationship Matrix (GRM):\n")
print(grm[1:6, 1:6])

# 检查 GRM 的维度
cat("GRM dimensions:", dim(grm), "\n")

# 可选：将 GRM 保存到文件
output_path <- paste(genofile,"_G.txt",sep="")
write.table(grm, output_path, row.names = FALSE, col.names = FALSE)
cat("GRM has been saved to:", output_path, "\n")