import numpy as np
from fire import Fire
import cv2
from pathlib import Path
import os


def read_mp4(fn):
    if not os.path.exists(fn):
        print(f"Filename {fn} does not exist")
    vidcap = cv2.VideoCapture(fn)
    frames = []
    while (vidcap.isOpened()):
        print(f"Frame")
        ret, frame = vidcap.read()
        if ret == False:
            break
        frames.append(frame)
    vidcap.release()
    print(f"Frames size {len(frames)}")
    return frames



def main(
        filename='./stock_videos/camel.mp4',
        S=48,  # seqlen
        N=1024,  # number of points per clip
        stride=8,  # spatial stride of the model
        timestride=1,  # temporal stride of the model
        iters=16,  # inference steps of the model
        image_size=(512, 896),  # input resolution
        max_iters=4,  # number of clips to run
        shuffle=False,  # dataset shuffling
        log_freq=1,  # how often to make image summaries
        log_dir='./logs_demo',
        init_dir='./reference_model',
        device_ids=[0],
):
    # the idea in this file is to run the model on a demo video,
    # and return some visualizations

    exp_name = 'de00'  # copy from dev repo

    print('filename', filename)
    name = Path(filename).stem
    print('name', name)
    print('dir', os.getcwd())

    rgbs = read_mp4(filename)
    rgbs = np.stack(rgbs, axis=0)  # S,H,W,3
    rgbs = rgbs[:, :, :, ::-1].copy()  # BGR->RGB
    rgbs = rgbs[::timestride]
    S_here, H, W, C = rgbs.shape
    print('rgbs', rgbs.shape)


if __name__ == '__main__':
    Fire(main)
