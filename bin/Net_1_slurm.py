# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras import Input, Model
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import numpy as np
import os
import keras
import glob
import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted
from sklearn.metrics import mean_squared_error
import time
import pathlib
from numpy import mean

# SLURM ��������
SLURM_TASK_ID = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))  # ��ǰ���� ID
SLURM_JOB_ID = os.getenv("SLURM_JOB_ID", "unknown_job")     # ��ǰ��ҵ ID

# Set CUDA_VISIBLE_DEVICES to -1 since GPU is unavailable
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load Processed Data
DigitalInput_fileList = natsorted(glob.glob("SplitGenotypeData_OneHot/" + "*.csv"))
num_files = len(DigitalInput_fileList)
num_tasks = int(os.getenv("SLURM_ARRAY_TASK_MAX", "1")) + 1  # ��������

# ƽ�������ļ�������
files_per_task = num_files // num_tasks  # ÿ���������С�ļ���
extra_files = num_files % num_tasks      # ������ļ���

# ���㵱ǰ�������ֹ����
start_idx = SLURM_TASK_ID * files_per_task + min(SLURM_TASK_ID, extra_files)
end_idx = start_idx + files_per_task + (1 if SLURM_TASK_ID < extra_files else 0)

# �����ļ��Ӽ�
task_file_list = DigitalInput_fileList[start_idx:end_idx]

print(f"Task {SLURM_TASK_ID}: Processing files {start_idx} to {end_idx}")
print(f"Files assigned: {task_file_list}")

# ��ʼѵ��
start_time = time.time()
evaluate = []
mse_loss = []
index = start_idx

for DigitalInput_filename in task_file_list:
    digital_data = pd.read_csv(DigitalInput_filename, low_memory=False, header=None)
    digital_data = np.array(digital_data)

    # ����ѵ���������Լ�����֤��
    x_train, x_mid = train_test_split(digital_data, test_size=0.4)
    x_test, x_valid = train_test_split(x_mid, test_size=0.5)
    print('Shape of train data: ', x_train.shape)
    print('Shape of test data: ', x_test.shape)
    print('Shape of validation data: ', x_valid.shape)

    # ���������
    input = Input(shape=(digital_data.shape[1],))
    print('Shape of input layer data: ', index, input.shape)

    # ����������ͽ�����
    encoded = Dense(120, activation='relu')(input)
    encoded = Dense(72, activation='relu')(encoded)
    encoded = Dense(10, activation='sigmoid')(encoded)

    decoded = Dense(72, activation='relu')(encoded)
    decoded = Dense(120, activation='relu')(decoded)
    decoded = Dense(digital_data.shape[1], activation='sigmoid')(decoded)

    # �����Զ�������ģ��
    autoencoder = Model(input, decoded)
    encoder = Model(input, encoded)

    autoencoder.summary()
    encoder.summary()

    # �����Ż�������ʧ����
    adm = keras.optimizers.Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=adm, loss='mse')

    # ������ͣ�ص�
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # ѵ��ģ��
    history = autoencoder.fit(x_train, x_train,
                              epochs=200,
                              batch_size=52,
                              shuffle=True,
                              validation_data=(x_valid, x_valid),
                              callbacks=[early_stopping])

    # Ԥ��ͱ�����
    enc_out = encoder.predict(digital_data)
    dec_out = autoencoder.predict(digital_data)
    
    output_dir = f'Net_1_EncData'
    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(f'{output_dir}/Net_1_EncData_{index}.csv', enc_out, delimiter=",", fmt='%1.0f')

    decoded_dir = f'Net_1_DecData'
    os.makedirs(decoded_dir, exist_ok=True)
    np.savetxt(f'{decoded_dir}/Net_1_DecData_{index}.csv', dec_out, delimiter=",", fmt='%1.0f')

    # ����ģ��
    model_dir = f'Net_1_Model'
    os.makedirs(model_dir, exist_ok=True)
    filename = f'{model_dir}/Net_1_Model_{index}.h5'
    autoencoder.save(filename)

    # ����ģ��
    score = autoencoder.evaluate(x_test, x_test, verbose=0)
    score_loss = score * 100
    evaluate.append([score_loss])

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    mse = mean_squared_error(digital_data, dec_out)
    print('MSE Loss: ', mse)
    mse_loss.append([mse])

    # ������ʧ��ʷ
    plot_dir = f'Net_1_Plot'
    os.makedirs(plot_dir, exist_ok=True)
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(f'{plot_dir}/Net_1_PlotLoss_{index}.png')
    plt.clf()

    index += 1

# ���������� MSE ���
evaluate_dir = 'Net_1_Evaluate'
os.makedirs(evaluate_dir, exist_ok=True)
np.savetxt(f'{evaluate_dir}/Net_1_Evaluate_{SLURM_TASK_ID}.csv', evaluate, delimiter=',', fmt='%1.3f', comments='', header='Loss')

mse_dir = 'Net_1_MSE'
os.makedirs(mse_dir, exist_ok=True)
np.savetxt(f'{mse_dir}/Net_1_MSE_{SLURM_TASK_ID}.csv', mse_loss, delimiter=',', fmt='%1.5f', comments='', header='MSE Loss')

mean_mse_loss = np.mean(mse_loss)
pathlib.Path(f"{mse_dir}/Net_1_MSE_{SLURM_TASK_ID}.txt").write_text(f"Task {SLURM_TASK_ID} MSE Loss: {mean_mse_loss}")

# ����ѵ��ʱ��
time_dir = "Net_1_Time"
os.makedirs(time_dir, exist_ok=True)
pathlib.Path(f"{time_dir}/Net_1_TrainingTime_{SLURM_TASK_ID}.txt").write_text(f"Task {SLURM_TASK_ID} Training Time: {time.time() - start_time}")

# ��ӡ���ս��
print("MSE Loss: ", mean(mse_loss))
print('Total Training Time: ', time.time() - start_time)