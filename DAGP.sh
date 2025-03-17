#!/bin/bash
#########################################################################
# 								
# Function: This script executes the DAGP method, including genotype one-hot encoding, data division, data combining, and genomic prediction using GBLUP, Bayesian, and machine learning methods. 									
# Written by Hailiang Song
# Copyright:  Beijing Academy of Agriculture and Forestry Sciences. All rights reserved.
# email: songhl0317@163.com									
# First version: 2024-12-6 
# 								
#########################################################################

#' @examples
#' ## DAGP Example
#' ./DAGP.sh pruned.raw id.txt rel.txt val.txt
#' pruned.raw      # PLINK-format genotype file encoded in 012 (0=homozygous reference, 1=heterozygous, 2=homozygous alternate).
#' id.txt          # A single-column file listing sample IDs in the same order as pruned.raw.
#' rel.txt         # Contains reference population IDs paired with their corresponding phenotypic values.
#' val.txt         # Includes validation population IDs and their associated phenotypes for evaluation.
            


SNP=$1
ID=$2
REL=$3
VAL=$4


#One-hot encoding and data division, split 1000 files.
python ./bin/DataProcessing012_fast.py ${SNP} 1000

#Deep autoencoder compression, multiple compression can be performed as required
python ./bin/Net_1_slurm.py
#Parallel compress chunks
#sbatch ./bin/jobs_Net_1.sh

#Combine compressed file
python ./bin/combine.py

#Calculating G and G Inverse Matrices for GBLUP method
Rscript ./bin/G.R Net_1_CompressedData.csv
python ./bin/Gma_to_3lineID.py Net_1_CompressedData.csv_G.txt ${ID} Gma
python ./bin/GINV.py Net_1_CompressedData.csv_G.txt Net_1_CompressedData.csv_G.txt_GINV
python ./bin/Gma_to_3lineID.py Net_1_CompressedData.csv_G.txt_GINV ${ID} Ginv

#GBLUP using compressed data
awk '{print $1,"1",$2}' ${REL} >relphe.txt
awk '{print $1,"1",$2}' ${VAL} >valphe.txt
cp ./bin/GBLUP.DIR ./
./bin/r_dmuai GBLUP		
python ./bin/gebv.py GBLUP.SOL valphe.txt yc_gebv_ld
python ./bin/COR_REG_used2.py yc_gebv_ld GBLUP.txt 1

#Bayesian methods
python ./bin/pickdata2.py ${ID} Net_1_CompressedData.csv ${VAL} val_fina.dat
python ./bin/pickdata2.py ${ID} Net_1_CompressedData.csv ${REL} rel_fina.dat	

#BayesB
cat rel_fina.dat val_fina.dat > all.dat
awk '{print $1,"NA"}' ${VAL} > lin.txt
cat rel.txt lin.txt > bayes.dat
Rscript ./bin/BayesB.R bayes.dat all.dat bayesB_gebv
paste bayes.dat bayesB_gebv > bayesBGEBV
python ./bin/picktbv.py ${VAL} bayesBGEBV val_gebv2.txt
python ./bin/COR_REG_used2.py val_gebv2.txt BayesB.txt 	1

#BayesA
Rscript ./bin/BayesA.R bayes.dat all.dat bayesB_gebv
paste bayes.dat bayesB_gebv > bayesBGEBV
python ./bin/picktbv.py ${VAL} bayesBGEBV val_gebv2.txt
python ./bin/COR_REG_used2.py val_gebv2.txt BayesA.txt 1	

#BayesCpi
Rscript ./bin/BayesC.R bayes.dat all.dat bayesB_gebv
paste bayes.dat bayesB_gebv > bayesBGEBV
python ./bin/picktbv.py ${VAL} bayesBGEBV val_gebv2.txt
python ./bin/COR_REG_used2.py val_gebv2.txt BayesCpi.txt 1

#BayesLasso
Rscript ./bin/BayesBL.R bayes.dat all.dat bayesB_gebv
paste bayes.dat bayesB_gebv > bayesBGEBV
python ./bin/picktbv.py ${VAL} bayesBGEBV val_gebv2.txt
python ./bin/COR_REG_used2.py val_gebv2.txt BayesLasso.txt 1

#Machine learning methods
python ./bin/CommML.py --rel_geno_file rel_fina.dat --rel_phe_file ${REL} --val_geno_file val_fina.dat --val_phe_file ${VAL} --model SVR
python ./bin/CommML.py --rel_geno_file rel_fina.dat --rel_phe_file ${REL} --val_geno_file val_fina.dat --val_phe_file ${VAL} --model RF 
python ./bin/CommML.py --rel_geno_file rel_fina.dat --rel_phe_file ${REL} --val_geno_file val_fina.dat --val_phe_file ${VAL} --model KRR
python ./bin/CommML.py --rel_geno_file rel_fina.dat --rel_phe_file ${REL} --val_geno_file val_fina.dat --val_phe_file ${VAL} --model XGB


python ./bin/COR_REG_used2.py val_gebv_SVR        SVR.txt        1
python ./bin/COR_REG_used2.py val_gebv_RF         RF.txt         1
python ./bin/COR_REG_used2.py val_gebv_KRR        KRR.txt        1
python ./bin/COR_REG_used2.py val_gebv_XGB        XGB.txt        1

