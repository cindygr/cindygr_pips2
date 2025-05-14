import time
import numpy as np
import saverloader
from nets.pips2 import Pips
import utils.improc
from utils.basic import print_, print_stats
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from fire import Fire
import sys
import cv2
from pathlib import Path
import os

def read_mp4(fn, image_size):
    if not os.path.exists(fn):
        print(f"Filename {fn} does not exist")
    vidcap = cv2.VideoCapture(fn)
    frames = []
    n_start = 0
    n_cap = 100
    while(vidcap.isOpened() and n_cap > 0):
        ret, frame = vidcap.read()
        width_start = (frame.shape[1] - image_size[1]) // 2
        height_start = (frame.shape[0] - image_size[0]) // 2
        width_end = width_start + image_size[1]
        height_end = height_start + image_size[0]
        if ret == False:
            break
        if n_start > 0:
            n_start -= 1
        else:
            frame_small = frame[height_start:height_end, width_start:width_end, :]
            #frame.resize((image_size[0], image_size[1], 3))
            frames.append(frame_small)
            n_cap -= 1
    vidcap.release()
    print(f"Frames size {len(frames)}")
    return frames

def run_model(model, rgbs, S_max=128, N=64, iters=16, sw=None, dev=None):
    rgbs = rgbs.to(dev).float() # B, S, C, H, W

    B, S, C, H, W = rgbs.shape
    assert(B==1)

    # pick N points to track; we'll use a uniform grid
    N_ = np.sqrt(N).round().astype(np.int32)
    grid_y, grid_x = utils.basic.meshgrid2d(B, N_, N_, stack=False, norm=False, device=dev)
    grid_y = 8 + grid_y.reshape(B, -1)/float(N_-1) * (H-16)
    grid_x = 8 + grid_x.reshape(B, -1)/float(N_-1) * (W-16)
    xy0 = torch.stack([grid_x, grid_y], dim=-1) # B, N_*N_, 2
    _, S, C, H, W = rgbs.shape

    # zero-vel init
    trajs_e = xy0.unsqueeze(1).repeat(1,S,1,1)

    iter_start_time = time.time()
    
    preds, preds_anim, _, _ = model(trajs_e, rgbs, iters=iters, feat_init=None, beautify=True)
    trajs_e = preds[-1]

    iter_time = time.time()-iter_start_time
    print('inference time: %.2f seconds (%.1f fps)' % (iter_time, S/iter_time))

    if sw is not None and sw.save_this:
        rgbs_prep = utils.improc.preprocess_color(rgbs)
        print("Writing video")
        sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='hot', linewidth=1, show_dots=True)
    return trajs_e


def main(
        filename='./stock_videos/tree_short.mov',
        S=48, # seqlen
        N=1024, # number of points per clip
        stride=8, # spatial stride of the model
        timestride=1, # temporal stride of the model
        iters=16, # inference steps of the model
        image_size=(512,896), # input resolution
        max_iters=4, # number of clips to run
        shuffle=False, # dataset shuffling
        log_freq=1, # how often to make image summaries
        log_dir='./logs_demo',
        init_dir='./reference_model',
        device_ids=[0],
):

    # the idea in this file is to run the model on a demo video,
    # and return some visualizations
    torch.set_default_dtype(torch.float)
    dev = "cpu"
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    torch.set_default_device(dev)

    print(f"Using device {dev}")
    exp_name = 'de00' # copy from dev repo

    print('filename', filename)
    name = Path(filename).stem
    print('name', name)
    print('dir', os.getcwd())

    rgbs = read_mp4(filename, image_size=(540, 960))
    rgbs = np.stack(rgbs, axis=0) # S,H,W,3
    rgbs = rgbs[:,:,:,::-1].copy() # BGR->RGB
    rgbs = rgbs[::timestride]
    S_here,H,W,C = rgbs.shape
    print('rgbs', rgbs.shape)

    # autogen a name
    model_name = "%s_%d_%d_%s" % (name, S, N, exp_name)
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)
    
    log_dir = 'logs_demo'
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    global_step = 0

    model = Pips(stride=8, device=dev).to(dev)
    parameters = list(model.parameters())
    if init_dir:
        print("fLoading {init_dir}")
        _ = saverloader.load(init_dir, model, dev=dev)
    global_step = 0
    model.eval()

    idx = list(range(0, max(S_here-S,1), S))
    #idx = list(range(100, 160, S))
    if max_iters:
        idx = idx[:max_iters]
    
    for si in idx:
        global_step += 1
        
        iter_start_time = time.time()

        sw_t = utils.improc.Summ_writer(
            writer=writer_t,
            global_step=global_step,
            log_freq=log_freq,
            fps=16,
            scalar_freq=int(log_freq/2),
            just_gif=True)

        rgb_seq = rgbs[si:si+S]
        rgb_seq = torch.from_numpy(rgb_seq).permute(0,3,1,2).to(torch.float32) # S,3,H,W
        rgb_seq = F.interpolate(rgb_seq, image_size, mode='bilinear').unsqueeze(0) # 1,S,3,H,W
        
        with torch.no_grad():
            trajs_e = run_model(model, rgb_seq, S_max=S, N=N, iters=iters, sw=sw_t, dev=dev)

        iter_time = time.time()-iter_start_time
        
        print('%s; step %06d/%d; itime %.2f' % (
            model_name, global_step, max_iters, iter_time))
        
            
    writer_t.close()
    print("Closed")

if __name__ == '__main__':
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print(x)
    else:
        print("MPS device not found.")
    Fire(main)
