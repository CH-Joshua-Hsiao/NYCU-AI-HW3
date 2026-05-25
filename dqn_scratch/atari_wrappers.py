import cv2
import numpy as np

def preprocess_frame(frame):
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized

class FrameStack:
    def __init__(self, n_frames=4):
        self.n_frames = n_frames
        self.frames = []

    def reset(self, obs):
        processed = preprocess_frame(obs)
        self.frames = [processed for _ in range(self.n_frames)]
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        processed = preprocess_frame(obs)
        self.frames.pop(0)
        self.frames.append(processed)
        return np.stack(self.frames, axis=0)
