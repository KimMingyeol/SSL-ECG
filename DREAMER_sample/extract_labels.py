import os
import numpy as np
import pandas as pd

### Sum and Save labels for each session of each participant
# NOTE: Session number starts by 1

name_map = {
    '최정원': 'j1',
    '윤승우': 'su',
    '이규식': 'ks',
    '이윤동': 'yd',
    '장재웅': 'jw'
}

score_map = {
    '거의 그렇지 않다.': 1,
    '가끔 그렇다.': 2,
    '자주 그렇다.': 3,
    '거의 언제나 그렇다.': 4
}

positive_question_num = [1, 2, 5, 8, 10, 11, 15, 16, 19, 20] # NOTE: Since question num. index starts by 1, you may have to adjust it layer

def score(question_num, answer):
    # print(question_num, answer)
    if answer not in score_map.keys():
        print("WARNING! Unexpected Answer:", answer, " at Question", question_num)
        return 0
    score = score_map[answer]
    return 5 - score if question_num in positive_question_num else score

labels_path = 'labels/stress_survey.csv'
labels = pd.read_csv(labels_path, header=0, sep=',')

output_root = 'labels_20231115'

session_num = 6
participants_session_stress_score = {}

for p_name in name_map.keys():
    print('---participant:', p_name)
    
    p_labels_np = labels.loc[labels['성함을 적어주세요.'] == p_name, :].to_numpy()
    p_labels_np = p_labels_np[:, 2:] # ignore timestamp(0), name(1) column
    p_labels_np = p_labels_np.reshape(session_num, -1)
    
    p_session_stress_score = []
    
    for session in range(session_num): # NOTE: Session num start by 1
        print("------session_num:", session+1)
        p_session = p_labels_np[session]
        p_stress_10 = p_session[0]
        p_stress_score = np.sum([score(question_num+1, answer) for question_num, answer in enumerate(p_session[1:])])
        
        p_session_stress_score.append({'session': session+1, 'stress_10': p_stress_10, 'stress_score': p_stress_score})
    
    participants_session_stress_score[name_map[p_name]] = p_session_stress_score

print(participants_session_stress_score)
stress_thr = 50

# Classify Stress & Non-Stress
# Generate DataFrame

dataframe_dict = {'name': []}
for session in range(session_num):
    dataframe_dict['sess{}'.format(session+1)] = []

for p, sessions in participants_session_stress_score.items():
    dataframe_dict['name'].append(p)
    for sess in sessions:
        dataframe_dict['sess{}'.format(sess['session'])].append('Stress' if sess['stress_score'] > stress_thr else 'Non-Stress')

os.makedirs(output_root, exist_ok=True)
df = pd.DataFrame(dataframe_dict)
df.to_csv(os.path.join(output_root, 'stress_labels.csv'), index=False)