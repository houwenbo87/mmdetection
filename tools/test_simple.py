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

#def single_gpu_test(model,
#                    data_loader,
#                    show=False,
#                    out_dir=None,
#                    show_score_thr=0.3):
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

    '''
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result, tuple):
            bbox_results, mask_results = result
            encoded_mask_results = encode_mask_results(mask_results)
            result = bbox_results, encoded_mask_results
        results.append(result)

        batch_size = len(data['img_metas'][0].data)
        for _ in range(batch_size):
            prog_bar.update()
    return results
    '''

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--list', help='image list')
    #parser.add_argument('--out_dir', help='output result file')
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
    outputs, wheel_pts = single_gpu_test(model, impaths, False, None, args.thresh)

    if args.out:
        print(f'\nwriting results to {args.out}')
        mmcv.dump(outputs, args.out)
        print(f'\nwriting results to {args.out}_pts.pkl')
        mmcv.dump(wheel_pts, args.out+'_pts.pkl')

if __name__ == '__main__':
    main()
