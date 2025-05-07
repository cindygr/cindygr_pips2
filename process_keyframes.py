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
import json
from pathlib import Path


def read_data(path_name):
    """get the N images needed for processing the points along with 2D points
    @param annot_name - annotation name
    @param kf - which keyframe
    @return images and points"""

    images = []
    for indx in range(0, 100):
        im_name = f"im{indx}.png"
        im = cv2.imread(path_name + im_name)
        if im is not None:
            images.append(im)
        else:
            break

    pts_2d = []
    with open(path_name + "pts.json", "r") as f:
        pts_2d = json.load(f)

    while len(images) > 8:
        images.remove(images[1])
    rgbs = np.stack(images)

    pts = np.array(pts_2d)

    return rgbs, pts


def run_model(model, rgbs, pts, iters=16, sw=None, dev=None):
    rgbs = rgbs.to(dev).float()  # B, S, C, H, W

    B, S, C, H, W = rgbs.shape
    assert (B == 1)


    # pick N points to track; we'll use a uniform grid
    N_ = np.sqrt(64).round().astype(np.int32)
    grid_y, grid_x = utils.basic.meshgrid2d(B, N_, N_, stack=False, norm=False, device=dev)
    grid_y = 8 + grid_y.reshape(B, -1) / float(N_ - 1) * (H - 16)
    grid_x = 8 + grid_x.reshape(B, -1) / float(N_ - 1) * (W - 16)
    xy0 = torch.stack([grid_x, grid_y], dim=-1)  # B, N_*N_, 2
    pts_as_np = np.array(pts)
    pts_xy = torch.tensor(pts_as_np, dtype=torch.float32).unsqueeze(0)
    _, S, C, H, W = rgbs.shape

    # zero-vel init
    trajs_e = pts_xy.unsqueeze(1).repeat(1, S, 1, 1)

    iter_start_time = time.time()

    preds, preds_anim, _, _ = model(trajs_e, rgbs, iters=iters, feat_init=None, beautify=True)
    trajs_e = preds[-1]

    iter_time = time.time() - iter_start_time
    print('inference time: %.2f seconds (%.1f fps)' % (iter_time, S / iter_time))

    if sw is not None and sw.save_this:
        rgbs_prep = utils.improc.preprocess_color(rgbs)
        sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]),
                                cmap='hot', linewidth=1, show_dots=False)
    return trajs_e


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
    annot_name = "first_tree_annot"
    path_start = "/Users/grimmc/"
    # path_start = "/Users/cindygrimm/"
    tree_path = path_start + "PycharmProjects/data/EnvyTree/"
    tree_name = "BP_R1_East_tree2"
    input_path = tree_path + tree_name + "/CalculatedData/pips2/input/"
    output_path = tree_path + tree_name + "/CalculatedData/pips2/output/"

    rgbs, pts = read_data(input_path)
    # the idea in this file is to run the model on a demo video,
    # and return some visualizations

    torch.set_default_tensor_type(torch.FloatTensor)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    exp_name = 'de00'  # copy from dev repo

    S_here, H, W, C = rgbs.shape
    print('rgbs', rgbs.shape)

    # autogen a name
    model_name = "%s_%d_%d_%s" % (annot_name, S, N, exp_name)
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)

    log_dir = 'logs_demo'
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    global_step = 0

    model = Pips(stride=8).to(dev)
    parameters = list(model.parameters())
    if init_dir:
        _ = saverloader.load(init_dir, model)
    global_step = 0
    model.eval()

    idx = list(range(0, max(S_here - S, 1), S))
    if max_iters:
        idx = idx[:max_iters]

    global_step += 1

    iter_start_time = time.time()

    sw_t = utils.improc.Summ_writer(
        writer=writer_t,
        global_step=global_step,
        log_freq=log_freq,
        fps=16,
        scalar_freq=int(log_freq / 2),
        just_gif=True)

    # Do the whole sequence
    rgb_seq = rgbs
    rgb_seq = torch.from_numpy(rgb_seq).permute(0, 3, 1, 2).to(torch.float32)  # S,3,H,W
    rgb_seq = F.interpolate(rgb_seq, image_size, mode='bilinear').unsqueeze(0)  # 1,S,3,H,W

    with torch.no_grad():
        trajs_e = run_model(model, rgb_seq, pts=pts, iters=iters, sw=None, dev=dev)
        pts_all = []
        for im_indx in range(0, trajs_e.shape[1]):
            pts_all.append([])
            for ipt in range(0, trajs_e.shape[2]):
                x = trajs_e[0][im_indx][ipt][0]
                y = trajs_e[0][im_indx][ipt][1]
                pts_all[-1].append([float(x), float(y)])
        fname_out = output_path + "pts_2d.json"
        with open(fname_out, "w") as f:
            json.dump(pts_all, f, indent=2)

    iter_time = time.time() - iter_start_time

    print('%s; step %06d/%d; itime %.2f' % (
        model_name, global_step, max_iters, iter_time))

    writer_t.close()


if __name__ == '__main__':
    Fire(main)
