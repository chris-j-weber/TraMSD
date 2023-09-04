import cv2
import random
import numpy as np
import torch

class VideoCapture:

    @staticmethod
    def load_frames_from_video(video_path: str, num_frames: int, video_sample_type: str):
        capture = cv2.VideoCapture(video_path)
        video_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        acc_samples = min(num_frames, video_length)
        intervals = np.linspace(0, video_length, acc_samples+1)
        range = []

        for idx, interval in enumerate(intervals[:-1]):
            range.append((interval, intervals[idx+1]-1))
        frame_ids = [(x[0] + x[1]) // 2 for x in range]

        frames = []
        # for i in range(num_frames):
        #     cap = cv2.VideoCapture(video_path)
        #     assert (cap.isOpened()), video_path
        #     ret, frame = cap.read()
        #     if ret:
        #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #         frame = cv2.resize(frame, (224, 224))
        #         frames.append(frame)
        #     cap.release()
        # return frames
    
        for i in frame_ids:
            capture.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = capture.read()
            if not ret:
                n_tries = 5
                for _ in range(n_tries):
                    ret, frame = capture.read()
                    if ret:
                        break
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                frame = frame.permute(2, 0, 1)
                frames.append(frame)
            else:
                raise ValueError
            
        while len(frames) < num_frames:
            frames.append(frames[-1].clone())

        frames = torch.stack(frames).float()
        capture.release()
        return frames, frame_ids