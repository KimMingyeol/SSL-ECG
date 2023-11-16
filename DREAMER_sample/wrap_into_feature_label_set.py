import pandas as pd
import numpy as np
import os

date = '20231116'
features_root = os.path.join('data', 'ecg_feature', date) # {user_name}/{sess~}.npy
labels_dir = os.path.join('data', 'label', date)

task = 'anxiety' # 'stress' or 'anxiety'
labels = pd.read_csv(os.path.join(labels_dir, '{}_labels.csv'.format(task)), header=0, sep=',')

participant_list = [i for i in os.listdir(features_root) if os.path.isdir(os.path.join(features_root, i))]
session_num = 6

df_dict = {}
df_dict_init = False

video_feature_root = os.path.join('data', 'video_features')

for p in participant_list:
    features_path = os.path.join(features_root, p)
    p_labels = labels.loc[labels['name'] == p, :]
    # features_flist = os.listdir(features_path) # each file name may indicate each session in a form of 'sess{}'.format(num)
    
    for sess in range(1, session_num+1):
        video_feature_path = os.path.join(video_feature_root, p, str(sess))
        video_feature_fns = os.listdir(video_feature_path)
        video_feature_fns.sort(key = lambda x: int(x.split('.')[0].split('_')[-1]))
        print("Checking Sort result:", video_feature_fns)
        
        video_feature = []
        for vf_fn in video_feature_fns:
            vf = pd.read_csv(os.path.join(video_feature_path, vf_fn), header=None, sep=' ').to_numpy()[0] # shape: (768,)
            video_feature.append(vf)
        video_feature = np.array(video_feature)
        
        fn = 'sess{}.npy'.format(sess)
        # assert fn in features_flist
        p_sess_features = np.load(os.path.join(features_path, fn))
        p_sess_features = np.concatenate((p_sess_features, video_feature), axis=1)
        
        p_sess_label = p_labels.loc[:, 'sess{}'.format(sess)].to_numpy()
        # print("type(p_sess_label):", type(p_sess_label))
        p_sess_labels = np.full((p_sess_features.shape[0],), p_sess_label)
        
        if not df_dict_init:
            for i in range(p_sess_features.shape[1]):
                df_dict['feat{}'.format(i)] = []
            df_dict['target'] = []
            df_dict_init = True
        
        for i in range(p_sess_features.shape[1]):
            df_dict['feat{}'.format(i)] = np.concatenate((df_dict['feat{}'.format(i)], p_sess_features[:, i]))
        df_dict['target'] = np.concatenate((df_dict['target'], p_sess_labels))

output_dir = os.path.join('data', 'dataset', date)
output_fn = '{}_dataset.csv'.format(task)
os.makedirs(output_dir, exist_ok=True)

df = pd.DataFrame(df_dict)
df.to_csv(os.path.join(output_dir, output_fn), index=False)