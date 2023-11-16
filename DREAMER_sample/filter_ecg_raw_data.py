import os
from shutil import copyfile
# NOTE: Session number starts by 1

filter_config = {
    'j1': {
        'baseline': None,
        'start': 2,
        'invalid': [],
    },
    'su': {
        'baseline': 2,
        'start': 3,
        'invalid': [1],
    },
    'ks': {
        'baseline': 1,
        'start': 2,
        'invalid': [3, 7],
    },
    'yd': {
        'baseline': 1,
        'start': 2,
        'invalid': [],
    },
    'jw': {
        'baseline': 1,
        'start': 2,
        'invalid': [],
    },
}

raw_data_root = "data_20231115"
out_data_root = "data_20231115_filtered"
participants = filter_config.keys()

# Denote baseline index as 0
save_baseline = False

for p in participants:
    print("Participant:", p)
    raw_data_path = os.path.join(raw_data_root, p)
    fn_csv_list = [i for i in os.listdir(raw_data_path) if i.split('.')[-1] == 'csv']
    f_csv_num = len(fn_csv_list)
    cnt = 1
    
    baseline_idx = filter_config[p]['baseline']
    start_idx = filter_config[p]['start']
    invalid_idx_list = filter_config[p]['invalid']
    
    out_data_path = os.path.join(out_data_root, p)
    os.makedirs(out_data_path, exist_ok=True)
    
    for i in range(1, f_csv_num + 1):
        fn = '{}_Session{}_Shimmer_ecg_Calibrated_PC.csv'.format(p, i)
        assert fn in fn_csv_list
        
        if baseline_idx is not None and i == baseline_idx:
            if save_baseline:
                copyfile(os.path.join(raw_data_path, fn), os.path.join(out_data_path, 'sess0.csv'))
                print("{}  -->  {}".format(os.path.join(raw_data_path, fn), os.path.join(out_data_path, 'sess0.csv')))
        if i not in invalid_idx_list:
            if i >= start_idx:
                copyfile(os.path.join(raw_data_path, fn), os.path.join(out_data_path, 'sess{}.csv'.format(cnt)))
                print("{}  -->  {}".format(os.path.join(raw_data_path, fn), os.path.join(out_data_path, 'sess{}.csv'.format(cnt))))
                cnt += 1
    print("=======================")