# Copyright (c) OpenMMLab. All rights reserved.
# python '/code/hchang27/mmaction2/demo/demo_spatiotemporal_det.py'  '/files/pathml/aim2/videos/CP/Participant 9/P9_TUG.mp4'  P9_TUG.mp4  --device "cuda:0" 
import argparse
import copy as cp
import tempfile

import cv2
import csv
import mmcv
import mmengine
import numpy as np
import torch
from mmengine import DictAction
from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData
import ffmpeg
import os

from mmaction.apis import detection_inference
from mmaction.registry import MODELS
from mmaction.structures import ActionDataSample
from mmaction.utils import frame_extract, get_str_type
torch.cuda.empty_cache()
try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1

def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))


plate_blue = '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
plate_blue = plate_blue.split('-')
plate_blue = [hex2color(h) for h in plate_blue]
plate_green = '004b23-006400-007200-008000-38b000-70e000'
plate_green = plate_green.split('-')
plate_green = [hex2color(h) for h in plate_green]


def visualize(frames, annotations, tracking_ids, plate=plate_blue, max_num=5):
    """Visualize frames with predicted annotations and tracking IDs.

    Args:
        frames (list[np.ndarray]): Frames for visualization.
        annotations (list[list[tuple]]): The predicted results.
        tracking_ids (list[np.ndarray]): List of tracking IDs for each frame.
        plate (str): The plate used for visualization. Default: plate_blue.
        max_num (int): Max number of labels to visualize for a person box.
            Default: 5.
    Returns:
        list[np.ndarray]: Visualized frames.
    """

    assert max_num + 1 <= len(plate)
    plate = [x[::-1] for x in plate]
    frames_out = cp.deepcopy(frames)
    nf, na = len(frames), len(annotations)
    assert nf % na == 0
    nfpa = len(frames) // len(annotations)
    anno = None
    h, w, _ = frames[0].shape
    scale_ratio = np.array([w, h, w, h])
    
    for i in range(na):
        anno = annotations[i]
        if anno is None:
            continue
        frame_track_ids = tracking_ids[i] if tracking_ids is not None and i < len(tracking_ids) else None
        
        for j in range(nfpa):
            ind = i * nfpa + j
            frame = frames_out[ind]
            for ann_idx, ann in enumerate(anno):
                box = ann[0]
                label = ann[1]
                if not len(label):
                    continue
                score = ann[2]
                box = (box * scale_ratio).astype(np.int64)
                st, ed = tuple(box[:2]), tuple(box[2:])
                cv2.rectangle(frame, st, ed, plate[0], 2)

                # Add tracking ID if available
                if frame_track_ids is not None and ann_idx < len(frame_track_ids):
                    track_id = frame_track_ids[ann_idx]
                    if track_id >= 0:  # Only show valid tracking IDs
                        id_text = f'ID: {track_id}'
                        id_location = (st[0], st[1] - 5)
                        
                        # Get text size for background box
                        (text_width, text_height), baseline = cv2.getTextSize(
                            id_text, FONTFACE, FONTSCALE, THICKNESS)
                        
                        # Draw white background rectangle
                        box_pt1 = (id_location[0], id_location[1] - text_height - baseline)
                        box_pt2 = (id_location[0] + text_width, id_location[1] + baseline)
                        cv2.rectangle(frame, box_pt1, box_pt2, (255, 255, 255), -1)
                        
                        # Draw black text
                        cv2.putText(frame, id_text, id_location, FONTFACE, FONTSCALE,
                                    (0, 0, 0), THICKNESS, LINETYPE)

                for k, lb in enumerate(label):
                    if k >= max_num:
                        break
                    text = abbrev(lb)
                    text = ': '.join([text, f'{score[k]:>.2f}'])
                    location = (0 + st[0], 18 + k * 18 + st[1])
                    textsize = cv2.getTextSize(text, FONTFACE, FONTSCALE,
                                               THICKNESS)[0]
                    textwidth = textsize[0]
                    diag0 = (location[0] + textwidth, location[1] - 14)
                    diag1 = (location[0], location[1] + 2)
                    cv2.rectangle(frame, diag0, diag1, plate[k + 1], -1)
                    cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                                FONTCOLOR, THICKNESS, LINETYPE)

    return frames_out

def log(frames, annotations, video, max_num=5, output_csv=None, fps=30, ids=None):
    """Log predicted annotations into a CSV file instead of visualizing.

    Args:
        frames (list[np.ndarray]): Frames used for predictions.
        annotations (list[list[tuple]]): The predicted results.
        video (str): Path to video file.
        max_num (int): Max number of labels to log for each person box.
        output_csv (str, optional): Path to output CSV file. If None, creates a new numbered file.
        fps (int): Frames per second of the video.
        ids (list[np.ndarray], optional): List of tracking IDs for each frame's detections.
    Returns:
        list[np.ndarray]: Original frames (unchanged).
    """
    frames_out = cp.deepcopy(frames)
    nf, na = len(frames), len(annotations)
    assert nf % na == 0
    nfpa = nf // na
    h, w, _ = frames[0].shape
    scale_ratio = np.array([w, h, w, h])

    base_filename = os.path.basename(video)
    folder_path = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(folder_path, "mmaction_result")
    os.makedirs(folder_path, exist_ok=True)

    # Use provided output CSV or create a new numbered file
    if output_csv:
        csv_filename = output_csv
    else:
        # Start with default name
        base_result_filename = 'results_'
        ext = '.csv'
        i = 0
        while True:
            csv_filename = os.path.join(folder_path, f'{base_result_filename}{i}{ext}')
            if not os.path.exists(csv_filename):
                break
            i += 1

    # Check if we need to write headers
    file_exists = os.path.exists(csv_filename)
    mode = 'a' if file_exists else 'w'
    
    with open(csv_filename, mode, newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        
        # Write headers only if file is new
        if not file_exists:
            writer.writerow(
                ['filename', 'fps', 'frames', 'track_id', 'detected_file'] + 
                [elem for i in range(max_num) for elem in [f'action_label_{i}', f'action_score_{i}']] + 
                ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']
            )

        for i in range(na):
            anno = annotations[i]
            if anno is None:
                continue
            
            frame_ids = ids[i] if ids is not None and i < len(ids) else None
            
            for j in range(nfpa):
                for ann_idx, ann in enumerate(anno):
                    box = ann[0]
                    labels = ann[1]
                    scores = ann[2]

                    if not len(labels):
                        continue

                    box = (box * scale_ratio).astype(np.int64)
                    x1, y1, x2, y2 = box

                    label_texts = []
                    for k in range(max_num):
                        if k >= len(labels):
                            label = "Not found"
                            score = -1
                        else:
                            label = labels[k]
                            score = scores[k]
                        label_texts.extend([label.replace(",", ";"), score])

                    track_id = frame_ids[ann_idx] if frame_ids is not None and ann_idx < len(frame_ids) else -1
                    frame_number = i * nfpa + j
                    writer.writerow([base_filename, fps, frame_number, track_id, base_filename] + label_texts + [x1, y1, x2, y2])
    
    print(f"Results {'appended to' if file_exists else 'written to'} {csv_filename}")
    return frames_out

def load_label_map(file_path):
    """Load Label Map.

    Args:
        file_path (str): The file path of label map.
    Returns:
        dict: The label map (int -> label name).
    """
    lines = open(file_path).readlines()
    lines = [x.strip().split(': ') for x in lines]
    return {int(x[0]): x[1] for x in lines}


def abbrev(name):
    """Get the abbreviation of label name:

    'take (an object) from (a person)' -> 'take ... from ...'
    """
    while name.find('(') != -1:
        st, ed = name.find('('), name.find(')')
        name = name[:st] + '...' + name[ed + 1:]
    return name


def pack_result(human_detection, result, img_h, img_w):
    """Short summary.

    Args:
        human_detection (np.ndarray): Human detection result.
        result (type): The predicted label of each human proposal.
        img_h (int): The image height.
        img_w (int): The image width.
    Returns:
        tuple: Tuple of human proposal, label name and label score.
    """
    human_detection[:, 0::2] /= img_w
    human_detection[:, 1::2] /= img_h
    results = []
    if result is None:
        return None
    for prop, res in zip(human_detection, result):
        res.sort(key=lambda x: -x[1])
        results.append(
            (prop.data.cpu().numpy(), [x[0] for x in res], [x[1]
                                                            for x in res]))
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument('video', help='video file/url')
    parser.add_argument(
        '--output-csv',
        help='Path to output CSV file. If not provided, will create a new numbered file.'
    )
    parser.add_argument(
        '--config',
        default=('configs/detection/slowonly/slowonly_kinetics400-pretrained-'
                 'r101_8xb16-8x8x1-20e_ava21-rgb.py'),
        help='spatialtemporal detection model config file path')
    parser.add_argument(
        '--checkpoint',
        default=('https://download.openmmlab.com/mmaction/detection/ava/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_'
                 '20201217-16378594.pth'),
        help='spatialtemporal detection model checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='/code/hchang27/mmaction2/configs/detection_configs/cascade_rcnn/cascade-mask-rcnn_x101-64x4d_fpn_20e_coco.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.7,
        help='the threshold of human detection score')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='the category id for human detection')
    parser.add_argument(
        '--action-score-thr',
        type=float,
        default=0.1,
        help='the threshold of human action score')
    parser.add_argument(
        '--label-map',
        default='tools/data/ava/label_map.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=384,
        help='specify the short-side length of the image')
    parser.add_argument(
        '--predict-stepsize',
        default=8,
        type=int,
        help='give out a prediction per n frames')
    parser.add_argument(
        '--output-stepsize',
        default=4,
        type=int,
        help=('show one frame per n frames in the demo, we should have: '
              'predict_stepsize % output_stepsize == 0'))
    parser.add_argument(
        '--output-fps',
        default=-1,
        type=int,
        help='the fps of demo video output')
    parser.add_argument(
        '--output-video',
        action='store_true',
        help='output video')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    args = parser.parse_args()
    return args
    

def main():
    args = parse_args()

    tmp_dir = tempfile.TemporaryDirectory()
    output_path = os.path.join(tmp_dir.name, 'rotated_video.mp4')
    ffmpeg.input(args.video).output(output_path, **{'metadata:s:v:0': 'rotate=0'}).overwrite_output().run()
    frame_paths, original_frames = frame_extract(output_path, out_dir=tmp_dir.name)
    num_frame = len(frame_paths)
    h, w, _ = original_frames[0].shape

    # resize frames to shortside
    new_w, new_h = mmcv.rescale_size((w, h), (args.short_side, np.Inf))
    frames = [mmcv.imresize(img, (new_w, new_h)) for img in original_frames]
    w_ratio, h_ratio = new_w / w, new_h / h

    # Get clip_len, frame_interval and calculate center index of each clip
    config = mmengine.Config.fromfile(args.config)
    config.merge_from_dict(args.cfg_options)
    val_pipeline = config.val_pipeline

    sampler = [
        x for x in val_pipeline if get_str_type(x['type']) == 'SampleAVAFrames'
    ][0]
    clip_len, frame_interval = sampler['clip_len'], sampler['frame_interval']
    window_size = clip_len * frame_interval
    assert clip_len % 2 == 0, 'We would like to have an even clip_len'
    # Note that it's 1 based here
    timestamps = np.arange(window_size // 2, num_frame + 1 - window_size // 2,
                           args.predict_stepsize)

    # Load label_map
    label_map = load_label_map(args.label_map)
    try:
        if config['data']['train']['custom_classes'] is not None:
            label_map = {
                id + 1: label_map[cls]
                for id, cls in enumerate(config['data']['train']
                                         ['custom_classes'])
            }
    except KeyError:
        pass

    # Get Human detection results
    center_frames = [frame_paths[ind - 1] for ind in timestamps]

    human_detections, _ = detection_inference(args.det_config,
                                            args.det_checkpoint,
                                            center_frames,
                                            args.det_score_thr,
                                            args.det_cat_id, 
                                            args.device,
                                            with_score=True)  # Enable scores and tracking
    torch.cuda.empty_cache()
    
    # Extract tracking IDs from the detection results
    tracking_ids = []
    for det in human_detections:
        if det.shape[1] > 5:  # If we have tracking IDs (bbox + score + track_id)
            tracking_ids.append(det[:, 5].astype(np.int32))  # Get track IDs
            det = det[:, :4]  # Keep only bbox coordinates for model input
        else:
            tracking_ids.append(np.full(det.shape[0], -1, dtype=np.int32))
        
    # Convert detections to tensor format for model input
    for i in range(len(human_detections)):
        det = human_detections[i][:, :4]  # Use only bbox coordinates
        det[:, 0:4:2] *= w_ratio
        det[:, 1:4:2] *= h_ratio
        human_detections[i] = torch.from_numpy(det).to(args.device)

    # Build STDET model
    try:
        # In our spatiotemporal detection demo, different actions should have
        # the same number of bboxes.
        config['model']['test_cfg']['rcnn'] = dict(action_thr=0)
    except KeyError:
        pass

    config.model.backbone.pretrained = None
    model = MODELS.build(config.model)

    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.to(args.device)
    model.eval()

    predictions = []

    img_norm_cfg = dict(
        mean=np.array(config.model.data_preprocessor.mean),
        std=np.array(config.model.data_preprocessor.std),
        to_rgb=False)

    print('Performing SpatioTemporal Action Detection for each clip')
    assert len(timestamps) == len(human_detections)
    prog_bar = mmengine.ProgressBar(len(timestamps))
    for timestamp, proposal in zip(timestamps, human_detections):
        if proposal.shape[0] == 0:
            predictions.append(None)
            continue

        start_frame = timestamp - (clip_len // 2 - 1) * frame_interval
        frame_inds = start_frame + np.arange(0, window_size, frame_interval)
        frame_inds = list(frame_inds - 1)
        imgs = [frames[ind].astype(np.float32) for ind in frame_inds]
        _ = [mmcv.imnormalize_(img, **img_norm_cfg) for img in imgs]
        # THWC -> CTHW -> 1CTHW
        input_array = np.stack(imgs).transpose((3, 0, 1, 2))[np.newaxis]
        input_tensor = torch.from_numpy(input_array).to(args.device)

        datasample = ActionDataSample()
        datasample.proposals = InstanceData(bboxes=proposal)
        datasample.set_metainfo(dict(img_shape=(new_h, new_w)))
        with torch.no_grad():
            result = model(input_tensor, [datasample], mode='predict')
            scores = result[0].pred_instances.scores
            prediction = []
            # N proposals
            for i in range(proposal.shape[0]):
                prediction.append([])
            # Perform action score thr
            for i in range(scores.shape[1]):
                if i not in label_map:
                    continue
                for j in range(proposal.shape[0]):
                    if scores[j, i] > args.action_score_thr:
                        prediction[j].append((label_map[i], scores[j,
                                                                   i].item()))
            predictions.append(prediction)
        prog_bar.update()

    results = []
    for human_detection, prediction in zip(human_detections, predictions):
        results.append(pack_result(human_detection, prediction, new_h, new_w))

    def dense_timestamps(timestamps, n):
        """Make it nx frames."""
        old_frame_interval = (timestamps[1] - timestamps[0])
        start = timestamps[0] - old_frame_interval / n * (n - 1) / 2
        new_frame_inds = np.arange(
            len(timestamps) * n) * old_frame_interval / n + start
        return new_frame_inds.astype(np.int64)

    dense_n = int(args.predict_stepsize / args.output_stepsize)
    frames = [
        cv2.imread(frame_paths[i - 1])
        for i in dense_timestamps(timestamps, dense_n)
    ]
    print('Performing visualization')
    vis_frames = visualize(frames, results, tracking_ids)
    # Calculate the exact fps needed for original duration
    video_capture = cv2.VideoCapture(args.video)
    original_fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    original_duration = total_frames / original_fps
    video_capture.release()
    if args.output_fps == -1:
        # Calculate exact fps needed to maintain original duration
        fps = len(vis_frames) / original_duration
    else:
        exact_fps = len(vis_frames) / original_duration
        requested_fps = args.output_fps
        
        # Find number of frames needed at requested fps to maintain duration
        ideal_frame_count = requested_fps * original_duration
        
        # If we need to drop frames
        if len(vis_frames) > ideal_frame_count:
            # Calculate step size to get as close as possible to requested fps
            step = len(vis_frames) / ideal_frame_count
            indices = [int(i * step) for i in range(int(ideal_frame_count))]
            vis_frames = [vis_frames[i] for i in indices]
            fps = requested_fps
        else:
            # If we don't have enough frames for requested fps, use exact fps
            fps = exact_fps
    
    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames],
                                fps=fps)
    base_filename = os.path.splitext(os.path.basename(args.video))[0]
    folder_path = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(folder_path, "mmaction_result")
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, base_filename+".mp4")
    if args.output_video:
        vid.write_videofile(file_path, codec="libx264")
    log(frames, results, args.video, max_num=5, output_csv=args.output_csv, fps=fps, ids=tracking_ids)

    tmp_dir.cleanup()


if __name__ == '__main__':
    main()
