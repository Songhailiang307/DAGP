import glob
import numpy as np
import pandas as pd
from natsort import natsorted

#load Net_1 Compressed Data
Net_1_Compressed_fileList = natsorted(glob.glob("Net_1_EncData/"
                                                +"*.csv"))
Net_1_Compressed_combined_csv = pd.concat([pd.read_csv(f, header=None)
                                           for f in Net_1_Compressed_fileList], axis=1)
print(Net_1_Compressed_combined_csv)
#export to csv
Net_1_Compressed_combined_csv.to_csv("Net_1_CompressedData.csv", index=False,
                                     header=None, float_format='%1.0f')

#load Net_2 Compressed Data
# Net_2_Compressed_fileList = natsorted(glob.glob("Net_2_EncData/"
                                                # +"*.csv"))
# Net_2_Compressed_combined_csv = pd.concat([pd.read_csv(f, header=None)
                                           # for f in Net_2_Compressed_fileList], axis=1)
# print(Net_2_Compressed_combined_csv)
# export to csv
# Net_2_Compressed_combined_csv.to_csv("Net_2_CompressedData.csv", index=False,
                                     # header=None, float_format='%1.0f')
# load Net_3 Compressed Data
# Net_3_Compressed_fileList = natsorted(glob.glob("Net_3_EncData/"
                                                # +"*.csv"))
# Net_3_Compressed_combined_csv = pd.concat([pd.read_csv(f, header=None)
                                           # for f in Net_3_Compressed_fileList], axis=1)
# print(Net_3_Compressed_combined_csv)
# export to csv
# Net_3_Compressed_combined_csv.to_csv("Net_3_CompressedData.csv", index=False,
                                     # header=None, float_format='%1.0f')
									 
# load Net_4 Compressed Data
# Net_4_Compressed_fileList = natsorted(glob.glob("Net_4_EncData/"
                                                # +"*.csv"))
# Net_4_Compressed_combined_csv = pd.concat([pd.read_csv(f, header=None)
                                           # for f in Net_4_Compressed_fileList], axis=1)
# print(Net_4_Compressed_combined_csv)
# export to csv
# Net_4_Compressed_combined_csv.to_csv("Net_4_CompressedData.csv", index=False,
                                     # header=None, float_format='%1.0f')

# load Net_5 Compressed Data
# Net_5_Compressed_fileList = natsorted(glob.glob("Net_5_EncData/"
                                                # +"*.csv"))
# Net_5_Compressed_combined_csv = pd.concat([pd.read_csv(f, header=None)
                                           # for f in Net_5_Compressed_fileList], axis=1)
# print(Net_5_Compressed_combined_csv)
# export to csv
# Net_5_Compressed_combined_csv.to_csv("Net_5_CompressedData.csv", index=False,
                                     # header=None, float_format='%1.0f')
# load Net_6 Compressed Data
# Net_6_Compressed_fileList = natsorted(glob.glob("Net_6_EncData/"
                                                # +"*.csv"))
# Net_6_Compressed_combined_csv = pd.concat([pd.read_csv(f, header=None)
                                           # for f in Net_6_Compressed_fileList], axis=1)
# print(Net_6_Compressed_combined_csv)
# export to csv
# Net_6_Compressed_combined_csv.to_csv("Net_6_CompressedData.csv", index=False,
                                     # header=None, float_format='%1.0f')

# load Net_7 Compressed Data
# Net_7_Compressed_fileList = natsorted(glob.glob("Net_7_EncData/"
                                                # +"*.csv"))
# Net_7_Compressed_combined_csv = pd.concat([pd.read_csv(f, header=None)
                                           # for f in Net_7_Compressed_fileList], axis=1)
# print(Net_7_Compressed_combined_csv)
# export to csv
# Net_7_Compressed_combined_csv.to_csv("Net_7_CompressedData.csv", index=False,
                                     # header=None, float_format='%1.0f')									 