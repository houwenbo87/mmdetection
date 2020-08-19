import argparse
import os
import os.path as osp
import shutil
import tempfile

import cv2
import math
import mmcv
import torch
import torch.nn as nn
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint

from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

from mmdet.core import auto_fp16, get_classes, tensor2imgs

from mmdet.utils.debugger import Debugger

import numpy as np

def nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat

def search_nearest(x, y, p, kp_xs, kp_ys, kp_clses):
    max_dis = 5
    for i, c in enumerate(kp_clses[0]):
        if p != c:
            continue
        if math.fabs(x - kp_xs[0][i]) > max_dis or math.fabs(y - kp_ys[0][i]) > max_dis:
            continue
        x = kp_xs[0][i]
        y = kp_ys[0][i]
        break
    return x,y

def single_gpu_test(img, model, show=False):

    results = []
    ori_img = cv2.resize(img, (352, 352))
    img = torch.FloatTensor(ori_img).permute(2, 0, 1).contiguous().unsqueeze(0)
    img[0, 0, :, :] = img[0, 0, :, :] - 102.9801
    img[0, 1, :, :] = img[0, 1, :, :] - 115.9465
    img[0, 2, :, :] = img[0, 2, :, :] - 122.7717

    img_meta = {'img_shape' : (352, 352, 3),  'scale_factor' : 1.0}
    outputs = model.forward_dummy(img)

    K = 6
    cls = outputs[0]
    hm = outputs[1]
    hm_kp = outputs[2]
    kps = outputs[3]
    kp_offset = outputs[4]

    cls = cls[0].detach().sigmoid().unsqueeze(0)
    hm = hm[0].detach().sigmoid().unsqueeze(0)
    kps = kps[0].detach().unsqueeze(0)
    hm_kp = hm_kp[0].detach().sigmoid().unsqueeze(0)
    kp_offset = kp_offset[0].detach().unsqueeze(0)

    featmap_sizes = [featmap.size()[-2:] for featmap in hm]

    batch, cat, height, width = hm.size()
    num_joints = kps.shape[1] // 2

    heat = nms(hm, kernel=11)
    #scores, clses = torch.max(heat, dim=0)
    scores, inds, clses, ys, xs = topk(heat, K=K)

    kps = transpose_and_gather_feat(kps, inds)
    kps = kps.view(batch, K, num_joints * 2)
    kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
    kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)
    xs = xs.view(batch, K, 1)
    ys = ys.view(batch, K, 1)
    scores = scores.view(batch, K, 1)

    hm_kp = nms(hm_kp, kernel=7)
    kp_scores, kp_inds, kp_clses, kp_ys, kp_xs = topk(hm_kp, K=K*4)
    kp_offset = transpose_and_gather_feat(kp_offset, kp_inds)
    kp_xs = kp_xs + kp_offset[..., 0]
    kp_ys = kp_ys + kp_offset[..., 1]

    #for k in range(K*4):
    #    s = kp_scores[0, k]
    #    if s < 0.4:
    #        continue
    #    x = int(kp_xs[0, k]) * 4
    #    y = int(kp_ys[0, k]) * 4
    #    cv2.circle(ori_img, (x,y), 2, (255,255,255), -1)

    cv2.rectangle(ori_img, (0,55), (359,305), (255,0,0))

    for k in range(K):
        s = scores[0, k, 0]
        if s < 0.36:
            continue
        xmin = 10000
        ymin = 10000
        xmax = 0
        ymax = 0
        ks = 0
        for p in range(4):
            x = int(kps[0, k, p*2])
            y = int(kps[0, k, p*2+1])
            x, y = search_nearest(x, y, p, kp_xs, kp_ys, kp_clses)
            x = min(featmap_sizes[0][1] - 1, x)
            y = min(featmap_sizes[0][0] - 1, y)
            x = max(0, x)
            y = max(0, y)

            if p == 0 or p == 3:
                k_s = hm_kp[0, p, int(y), int(x)]
                if (k_s > 0.4):
                    kps[0, k, p*2] = x
                    kps[0, k, p*2+1] = y
                ks += k_s
            xmin = min(xmin, x)
            ymin = min(ymin, y)
            xmax = max(xmax, x)
            ymax = max(ymax, y)
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        cl0 = cls[0, 0, int(cy), int(cx)]
        cl1 = cls[0, 1, int(cy), int(cx)]
        if cl0 > cl1:
            cat_l = 0
            clr = (0, 255, 0)
        else:
            cat_l = 1
            clr = (0, 0, 255)
        
        if max(cl0, cl1) * s < 0.3:
            continue

        if ks < 0.1:
            continue

        x0 = int(kps[0, k, 0]) * 4
        y0 = int(kps[0, k, 1]) * 4
        x1 = int(kps[0, k, 2]) * 4
        y1 = int(kps[0, k, 3]) * 4
        dis1 = math.sqrt((x0-x1)*(x0-x1) + (y0-y1)*(y0-y1))
        x2 = int(kps[0, k, 4]) * 4
        y2 = int(kps[0, k, 5]) * 4
        x3 = int(kps[0, k, 6]) * 4
        y3 = int(kps[0, k, 7]) * 4
        dis2 = math.sqrt((x2-x3)*(x2-x3) + (y2-y3)*(y2-y3))
        dis3 = math.sqrt((x0-x3)*(x0-x3) + (y0-y3)*(y0-y3))
        if max(dis1, dis2) < 40 and dis3 < 80:
            continue

        for p in range(3):
            x0 = int(kps[0, k, p*2]) * 4
            y0 = int(kps[0, k, p*2+1]) * 4
            x1 = int(kps[0, k, (p+1)*2]) * 4
            y1 = int(kps[0, k, (p+1)*2+1]) * 4
            cv2.line(ori_img, (x0,y0), (x1,y1), clr, 1)
        x0 = int(kps[0, k, 0]) * 4
        y0 = int(kps[0, k, 1]) * 4
        x1 = int(kps[0, k, 3*2]) * 4
        y1 = int(kps[0, k, 3*2+1]) * 4
        cv2.line(ori_img, (x0,y0), (x1,y1), clr, 1)
        results.append((cat_l, kps[0, k, 0]*4, kps[0, k, 1]*4, kps[0, k, 2]*4, kps[0, k, 3]*4, kps[0, k, 4]*4, kps[0, k, 5]*4, kps[0, k, 6]*4, kps[0, k, 7]*4))
        #str_info = '{}, {}, {}, {}'.format(s, cl0, cl1, ks)
        #cv2.putText(ori_img, str_info, (x0,y0), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 1, cv2.LINE_AA)
        #str_info = '{}, {}'.format(max(cl0, cl1), ks)
        #cv2.putText(ori_img, str_info, (x0,y0+20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 1, cv2.LINE_AA)
        #str_info = '{}'.format(ks)
        #cv2.putText(ori_img, str_info, (x0,y0+40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 1, cv2.LINE_AA)

    '''
    img_show = ori_img

    bbox_result, keypt_result = result

    bboxes = np.vstack(bbox_result)
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    mmcv.imshow_det_bboxes(
        img_show,
        bboxes,
        labels,
        score_thr=0.5,
        wait_time=10)

    results.append(result)
    '''

    return ori_img, results

def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--list', help='image list')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--json_out',
        help='output result file name without extension',
        type=str)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    assert args.out or args.show or args.json_out, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out" or "--show" or "--json_out"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if args.json_out is not None and args.json_out.endswith('.json'):
        args.json_out = args.json_out[:-5]
    
    if args.list is not None:
        with open(args.list) as f:
            impaths = [line.strip() for line in f]

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    model.eval()

    # for video
    video_id = [1, 5, 6, 7, 10]
    for id in video_id:
        print('process video: #{}'.format(id))
        in_video_name = '{}.avi'.format(id)
        out_video_name = 'parking_slot/UNTOUCH-parking_slot_{}.mp4'.format(id)
        videoCapture = cv2.VideoCapture(in_video_name)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        videoWriter = cv2.VideoWriter(out_video_name, fourcc, fps, size)

        logo_b = cv2.imread('logo.png')

        l_w = 125
        l_h = 38

        logo_b = cv2.resize(logo_b, (l_w, l_h), interpolation=cv2.INTER_CUBIC)

        success, frame = videoCapture.read()

        frame = cv2.resize(frame, (352, 352))
        image_height = frame.shape[0]
        image_width = frame.shape[1]
        logo_b = logo_b.astype(np.float32)

        idx = 0
        det_cnt = 0
        idle_cnt = 0
        while success:
            frame = frame.astype(np.float32)
            frame, results = single_gpu_test(frame, model, args.show)
            frame[image_height-l_h:, image_width-l_w:, :] = frame[image_height-l_h:, image_width-l_w:, :] + logo_b * 0.5
            frame = np.minimum(frame, 255)
            frame = frame.astype(np.uint8)
            frame = cv2.resize(frame, size)
            #cv2.imshow("Oto Video", frame)
            #cv2.waitKey(10)
            videoWriter.write(frame)
            cv2.imshow('lane spot', frame)
            cv2.waitKey(10)

            idx += 1
            imname = 'parking_slot/{}_{:04d}.jpg'.format(id, idx)
            cv2.imwrite(imname, frame)

            det_cnt += len(results)
            for r in results:
                if r[0] == 0:
                    idle_cnt += 1

            success, frame = videoCapture.read()
        print('detect num: {}'.format(det_cnt))
        print('idle num: {}'.format(idle_cnt))
        print('image num: {}'.format(idx))

    '''
    for i, impath in enumerate(impaths):
        img = cv2.imread(impath)
        if img is None:
            continue
        basename = os.path.basename(impath)
        outputs = single_gpu_test(img, model, args.show)
        cv2.imwrite(basename, outputs)
        cv2.imshow('parking slot', outputs)
        cv2.waitKey(10)
    '''

if __name__ == '__main__':
    main()
