import os
import numpy as np
import pandas as pd
from scipy import signal

# handling multiple users' data

def normalize(x, x_mean, x_std):
    """ 
    perform z-score normalization of a signal """
    x_scaled = (x-x_mean)/x_std
    return x_scaled

def make_window(signal, fs, overlap, window_size_sec):
    """ 
    perform cropped signals of window_size seconds for the whole signal
    overlap input is in percentage of window_size
    window_size is in seconds """
    
    window_size = fs * window_size_sec
    overlap     = int(window_size * (overlap / 100))
    start       = 0   
    segmented   = np.zeros((1, window_size), dtype = int)
    while(start+window_size <= len(signal)):
        segment     = signal[start:start+window_size]
        segment     = segment.reshape(1, len(segment))
        segmented   = np.append(segmented, segment, axis =0)
        start       = start + window_size - overlap
    return segmented[1:]

data_dir = 'data_20231115_filtered'
output_root = 'output_ecg_20231115'

fs = 256 ### no downsampling required for DREAMER format
order = 10 # order's not described in the paper
cut_off_freq = 0.8 # pass-band frequency described in the paper

window_size_sec = 10
window_size = window_size_sec * fs

sos = signal.butter(order, [cut_off_freq], 'highpass', fs=fs, output='sos')

participant_list = [i for i in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, i))]

for participant in participant_list:
    print('Processing', participant)
    
    fn_list = [i for i in os.listdir(os.path.join(data_dir, participant)) if i.split('.')[-1] == 'csv']
    output_dir = os.path.join(output_root, participant)
    os.makedirs(output_dir, exist_ok=True)
    
    filtered_ecg_dict = {}
    
    for fn in fn_list:
        data_path = os.path.join(data_dir, participant, fn)
        output_fn = fn.split('.')[0] + '.npy'
        
        data = pd.read_csv(data_path, header = 1, skiprows=[2], sep='\t')
        
        ecg = data.loc[:, 'Shimmer_ecg_ECG_LL-RA_24BIT_CAL'] # unit: mV
        timestamp = data.loc[:, 'Shimmer_ecg_TimestampSync_Unix_CAL'] # unit: ms
        
        ecg = ecg.to_numpy()
        timestamp = timestamp.to_numpy()
        
        ### STEP1: high-pass IIR filter
        filtered_ecg = signal.sosfilt(sos, ecg)
        filtered_ecg_dict[output_fn] = {'filtered_ecg': filtered_ecg.reshape(len(filtered_ecg), 1), 'timestamp': timestamp.reshape(len(timestamp), 1)}
        ###
    
    ### STEP2: user-specific z-score normalization
    filtered_ecg_list = [v['filtered_ecg'] for v in filtered_ecg_dict.values()]
    print('len(filtered_ecg_list)=', len(filtered_ecg_list), ', containing', filtered_ecg_list[0].shape)
    filtered_ecg_merged = np.vstack(filtered_ecg_list)
    print('shape of filtered_ecg_merged', filtered_ecg_merged.shape)
    
    filtered_ecg_merged_sorted = np.sort(filtered_ecg_merged)
    row_num = filtered_ecg_merged_sorted.shape[0]
    std = np.std(filtered_ecg_merged_sorted[np.int(0.025*row_num) : np.int(0.975*row_num)])
    mean = np.mean(filtered_ecg_merged_sorted)
    
    filtered_normalized_ecg_dict = {k: {'ecg': normalize(v['filtered_ecg'], mean, std), 'timestamp': v['timestamp']} for (k, v) in filtered_ecg_dict.items()}
    ###
    
    ### STEP3: segment into 10 seconds time window
    for (k, v) in filtered_normalized_ecg_dict.items():
        filtered_normalized_windowed_ecg = make_window(v['ecg'], fs, 0, window_size_sec)
        # print(filtered_normalized_windowed_ecg)
        print(filtered_normalized_windowed_ecg.shape)
        np.save(os.path.join(output_dir, k), filtered_normalized_windowed_ecg)
    ###