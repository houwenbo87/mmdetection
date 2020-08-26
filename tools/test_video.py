import argparse
import os
import cv2

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

def single_gpu_test_video(model,
                    video_path,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    videoCapture = cv2.VideoCapture(video_path)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    video_basename = os.path.basename(video_path)[0:-4]
    idx = 0

    if out_dir is not None:
        dstpath = os.path.join(out_dir, video_basename + '.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        videoWriter = cv2.VideoWriter(dstpath, fourcc, fps, size)

    # img preprocess
    bbox_results = []
    wheelpts_results = []
    prog_bar = mmcv.ProgressBar(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    success = True
    while success:
        idx += 1
        success, src_img = videoCapture.read()
        prog_bar.update()
        #src_img = cv2.imread(impath)
        if src_img is None:
            return bbox_results, wheelpts_results
        input_w = 1280
        input_h = 736
        scale_x = src_img.shape[1] / input_w
        scale_y = src_img.shape[0] / input_h
        img = cv2.resize(src_img, (input_w, input_h))
        img = torch.FloatTensor(img).permute(2, 0, 1).contiguous().unsqueeze(0)
        img[0, 0, :, :] = (img[0, 0, :, :] - 103.53) / 57.375
        img[0, 1, :, :] = (img[0, 1, :, :] - 116.28) / 57.12
        img[0, 2, :, :] = (img[0, 2, :, :] - 123.675) / 58.395
        img_meta = {'img_shape' : (input_h, input_w, 3),  'scale_factor' : 1.0}

        with torch.no_grad():
            #result = model(return_loss=False, rescale=True, **data)
            bboxes = model.module.simple_test(img.cuda(), img_metas=[img_meta])
            for cat in range(len(bboxes)):
                bboxes[cat][:, 0] *= scale_x
                bboxes[cat][:, 2] *= scale_x
                bboxes[cat][:, 1] *= scale_y
                bboxes[cat][:, 3] *= scale_y
        bbox_results.append(bboxes)

        clr = [(0,255,0), (0,0,255), (255,0,0), (255,255,0), (255,0,255), (128,255,0), (255,0,128), (0,0,255)]

        if out_dir is not None:
            dstpath = os.path.join(out_dir, video_basename + '_{:06d}'.format(idx) + '.jpg')
            for cat in range(len(bboxes)):
                for bbox in bboxes[cat]:
                    if bbox[-1] > show_score_thr:
                        cv2.rectangle(src_img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), clr[cat], 1)
            cv2.imwrite(dstpath, src_img)
            videoWriter.write(src_img)

    return bbox_results
 
def single_gpu_test(model,
                    impaths,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    # img preprocess
    bbox_results = []
    wheelpts_results = []
    prog_bar = mmcv.ProgressBar(len(impaths))
    for i, impath in enumerate(impaths):
        prog_bar.update()
        src_img = cv2.imread(impath)
        if src_img is None:
            return bbox_results, wheelpts_results
        input_w = 1280
        input_h = 736
        scale_x = src_img.shape[1] / input_w
        scale_y = src_img.shape[0] / input_h
        img = cv2.resize(src_img, (input_w, input_h))
        img = torch.FloatTensor(img).permute(2, 0, 1).contiguous().unsqueeze(0)
        img[0, 0, :, :] = (img[0, 0, :, :] - 103.53) / 57.375
        img[0, 1, :, :] = (img[0, 1, :, :] - 116.28) / 57.12
        img[0, 2, :, :] = (img[0, 2, :, :] - 123.675) / 58.395
        img_meta = {'img_shape' : (input_h, input_w, 3),  'scale_factor' : 1.0}

        with torch.no_grad():
            #result = model(return_loss=False, rescale=True, **data)
            bboxes, wheel_pts = model.module.simple_test_all(img.cuda(), img_metas=[img_meta])
            for cat in range(len(bboxes)):
                bboxes[cat][:, 0] *= scale_x
                bboxes[cat][:, 2] *= scale_x
                bboxes[cat][:, 1] *= scale_y
                bboxes[cat][:, 3] *= scale_y
            for pts in wheel_pts:
                pts[0] *= scale_x
                pts[1] *= scale_y
                #pts[2] *= scale_x
                #pts[3] *= scale_y
        bbox_results.append(bboxes)
        wheelpts_results.append(wheel_pts)
    return bbox_results, wheelpts_results

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--list', help='image list')
    parser.add_argument('--video', help='video path')
    parser.add_argument('--out_dir', help='output result file')
    parser.add_argument('--thresh',
                        type=float,
                        default=0.3,
                        help='score threshold (default: 0.3)')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if args.list is not None:
        with open(args.list) as f:
            impaths = [line.strip() for line in f]

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    distributed = False

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    #dataset = build_dataset(cfg.data.test)
    #data_loader = build_dataloader(
    #    dataset,
    #    samples_per_gpu=1,
    #    workers_per_gpu=cfg.data.workers_per_gpu,
    #    dist=distributed,
    #    shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    #fp16_cfg = cfg.get('fp16', None)
    #if fp16_cfg is not None:
    #    wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    #outputs = single_gpu_test(model, data_loader, False, None, args.thresh)
    if args.list is not None:
        outputs, wheel_pts = single_gpu_test(model, impaths, False, None, args.thresh)
    elif args.video is not None:
        outputs  = single_gpu_test_video(model, args.video, False, args.out_dir, args.thresh)

    if args.out:
        print(f'\nwriting results to {args.out}')
        mmcv.dump(outputs, args.out)

if __name__ == '__main__':
    main()
