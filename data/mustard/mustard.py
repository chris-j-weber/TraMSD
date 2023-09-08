import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('mustard++_text.csv')
df.drop(['END_TIME', 'SPEAKER', 'SHOW', 'Sarcasm_Type', 'Valence', 'Arousal'],axis=1,inplace=True)

keys = []

for index, row in df.iterrows():
    if row['Sarcasm'] in [0.0, 1.0]:
        tmp = {}
        tmp.update({'key': row['SCENE']})
        tmp.update({'image': row['KEY']})
        tmp.update({'label': row['Sarcasm']})
        keys.append(tmp)

scenes = []
for k in keys:
    sent = ''
    for index, row in df.iterrows():
        if k['key'] == row['SCENE']:
            sent += ' ' + row['SENTENCE']
    k.update({'sentence': sent})
    scenes.append(k)

final = pd.DataFrame(scenes)

train_val, test = train_test_split(final, test_size=0.1)
train, val = train_test_split(train_val, test_size=0.1)

train.to_csv('train_mustard.csv', index=False)
test.to_csv('test_mustard.csv', index=False)
val.to_csv('val_mustard.csv', index=False)

for sc in scenes:
    cam = cv2.VideoCapture('videos/final_utterance_videos/'+sc['image']+'.mp4')

    ret, frame = cam.read()
    if ret:
        cv2.imwrite('img/'+sc['image']+'.jpg', frame)

    cam.release()
    cv2.destroyAllWindows()