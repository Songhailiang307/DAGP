# -*- coding: utf-8 -*-
import sys
import numpy as np

# File paths
id_file = sys.argv[1]           # ID file path
genotype_file = sys.argv[2]     # Genotype file path
val_id_file = sys.argv[3]       # File with IDs to filter
output_file = sys.argv[4]       # Output file path

# Step 1: Read ID file
with open(id_file, 'r') as f:
    ids = [line.strip() for line in f]

# Step 2: Read genotype file (comma-separated)
with open(genotype_file, 'r') as f:
    genotypes = [line.strip().split(',') for line in f]

# Step 3: Convert genotype data to numpy array for variance computation
genotype_array = np.array(genotypes, dtype=float)  # Ensure numeric data for variance computation

# Step 4: Calculate standard deviation for each column and filter SNPs
initial_snp_count = genotype_array.shape[1]  # Total number of SNPs initially
column_std = np.std(genotype_array, axis=0)
valid_columns = column_std > 0  # Identify columns with non-zero variance
filtered_genotype_array = genotype_array[:, valid_columns]  # Filter SNPs

# Step 5: Convert filtered genotype array to integers
filtered_genotype_array = filtered_genotype_array.astype(int)  # Convert to integer type
final_snp_count = filtered_genotype_array.shape[1]  # Total number of SNPs after filtering

# Step 6: Print SNP filtering summary
filtered_snp_count = initial_snp_count - final_snp_count
print(f"Initial SNP count: {initial_snp_count}")
print(f"Final SNP count: {final_snp_count}")
print(f"Filtered SNP count: {filtered_snp_count}")

# Step 7: Replace original genotypes with filtered genotypes
genotypes = filtered_genotype_array.tolist()

# Step 8: Read val_id file and store in a set for fast lookup
with open(val_id_file, 'r') as f:
    val_ids = [line.strip() for line in f]

# Step 9: Check if ID file and genotype file have the same number of lines
if len(ids) != len(genotypes):
    raise ValueError("The number of lines in the ID file and genotype file do not match. Please check the data.")

# Step 10: Create a dictionary for fast lookups of genotype by ID
id_to_genotype = {ids[i]: genotypes[i] for i in range(len(ids))}

# Step 11: Filter the genotypes using val_ids and write to output file (keeping val_id_file order)
with open(output_file, 'w') as f:
    for val_id in val_ids:
        val_id = val_id.split()[0]  # Handle the val_id file format (if it contains extra spaces or lines)
        if val_id in id_to_genotype:
            genotype = id_to_genotype[val_id]
            f.write(f"{val_id} {' '.join(map(str, genotype))}\n")

print(f"Filtered data saved to: {output_file}")