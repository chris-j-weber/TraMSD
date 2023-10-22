import cv2
import random
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
final['label'] = final['label'].astype(int)

train_val, test = train_test_split(final, test_size=0.1)
train, val = train_test_split(train_val, test_size=0.1)

train.to_csv('train_mustard.csv', index=False)
test.to_csv('test_mustard.csv', index=False)
val.to_csv('val_mustard.csv', index=False)

num_frames = 4
for sc in scenes:
  vidpath = 'data/mustard/videos/final_utterance_videos/'+sc['image']+'.mp4'
  cam = cv2.VideoCapture(vidpath)
  total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
  video_id = sc['image']

  acc_samples = min(num_frames, total_frames)

  random_frame_idxs = random.sample(range(total_frames), acc_samples)

  for frame_idx in sorted(random_frame_idxs):
    cam.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    ret, frame = cam.read()
    if not ret:
      print(f'frame {frame_idx} could not be read')
      continue

    output_filename = f'data/mustard/frames/{video_id}_{frame_idx}.jpg'
    cv2.imwrite(output_filename, frame)

  cam.release()
  cv2.destroyAllWindows()