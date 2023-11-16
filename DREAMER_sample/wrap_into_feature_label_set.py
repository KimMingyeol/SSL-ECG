import pandas as pd
import numpy as np
import os

features_root = "output_feature_20231115" # {user_name}/{sess~}.npy
labels_dir = "labels_20231115"

labels = pd.read_csv(os.path.join(labels_dir, 'stress_labels.csv'), header=0, sep=',')

participant_list = [i for i in os.listdir(features_root) if os.path.isdir(os.path.join(features_root, i))]
session_num = 6

df_dict = {}
df_dict_init = False

for p in participant_list:
    features_path = os.path.join(features_root, p)
    p_labels = labels.loc[labels['name'] == p, :]
    # features_flist = os.listdir(features_path) # each file name may indicate each session in a form of 'sess{}'.format(num)
    for sess in range(1, session_num+1):
        fn = 'sess{}.npy'.format(sess)
        # assert fn in features_flist
        p_sess_features = np.load(os.path.join(features_path, fn))
        p_sess_label = p_labels.loc[:, 'sess{}'.format(sess)].to_numpy()
        
        p_sess_binary_label = 1 if p_sess_label == 'Stress' else 0
        p_sess_labels = np.full((p_sess_features.shape[0],), p_sess_binary_label)
        
        if not df_dict_init:
            for i in range(p_sess_features.shape[1]):
                df_dict['feat{}'.format(i)] = []
            df_dict['target'] = []
            df_dict_init = True
        
        for i in range(p_sess_features.shape[1]):
            df_dict['feat{}'.format(i)] = np.concatenate((df_dict['feat{}'.format(i)], p_sess_features[:, i]))
        df_dict['target'] = np.concatenate((df_dict['target'], p_sess_labels))

output_dir = 'dataset'
output_fn = 'dataset_20231115.csv'
os.makedirs(output_dir, exist_ok=True)

df = pd.DataFrame(df_dict)
df.to_csv(os.path.join(output_dir, output_fn), index=False)