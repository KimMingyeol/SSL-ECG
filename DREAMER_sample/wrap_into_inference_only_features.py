import pandas as pd
import numpy as np
import os

date = '20231116'
features_root = os.path.join('data', 'ecg_feature', date) # {user_name}/{sess~}.npy
output_root = os.path.join('data', 'inference_only', date)

participant_list = [i for i in os.listdir(features_root) if os.path.isdir(os.path.join(features_root, i))]
session_list = [7, 8 ,9]

video_feature_root = os.path.join('data', 'video_features')

for p in participant_list:
    features_path = os.path.join(features_root, p)
    # features_flist = os.listdir(features_path) # each file name may indicate each session in a form of 'sess{}'.format(num)
    
    for sess in session_list:
        df_dict = {}
    
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
        
        for i in range(p_sess_features.shape[1]):
            df_dict['feat{}'.format(i)] = []
        
        for i in range(p_sess_features.shape[1]):
            df_dict['feat{}'.format(i)] = np.concatenate((df_dict['feat{}'.format(i)], p_sess_features[:, i]))
        
        output_path = os.path.join(output_root, p)
        output_fn = 'sess{}.csv'.format(sess)
        os.makedirs(output_path, exist_ok=True)
        
        df = pd.DataFrame(df_dict)
        df.to_csv(os.path.join(output_path, output_fn), index=False)