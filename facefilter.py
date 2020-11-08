import os
import cv2
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms

from face_detection_dsfd.face_ssd_infer import SSD
from face_detection_dsfd.data import widerface_640, TestBaseTransform

from fsgan.datasets.appearance_map import fuse_clusters
from fsgan.datasets import img_lms_pose_transforms, img_landmarks_transforms
from fsgan.datasets.img_lms_pose_transforms import RandomHorizontalFlip, Rotate, Pyramids, ToTensor, Normalize, Resize
from fsgan.inference.swap import transfer_mask, select_seq
from fsgan.preprocess.preprocess_video import smooth_poses
from fsgan.utils.bbox_utils import batch_iou, scale_bbox, crop_img, crop2img, smooth_bboxes
from fsgan.utils.img_utils import create_pyramid
from fsgan.utils.landmarks_utils import LandmarksHeatMapEncoder, LandmarksHeatMapDecoder, smooth_landmarks_98pts
from fsgan.utils.obj_factory import obj_factory
from fsgan.utils.seg_utils import SoftErosion, remove_inner_mouth
from fsgan.utils.temporal_smoothing import TemporalSmoothing
from fsgan.utils.utils import set_device, load_model
from fsgan.utils.video_utils import Sequence

torch.set_grad_enabled(False)

HALF = False

float_type = 'float32'
torch.set_default_tensor_type(torch.FloatTensor)
if HALF:
    float_type = 'float16'
    # torch.set_default_tensor_type(torch.HalfTensor)
    # torch.set_default_tensor_type('torch.HalfTensor')

# base_path = 'C:/data/dev/projects/fsgan'
base_path = 'fsgan'
batch_size = 1
det_batch_size = 1
finetune_batch_size = 1

# preprocess_video
base_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)

general = base_parser.add_argument_group('general')
general.add_argument('-r', '--resolution', default=256, type=int, metavar='N',
                     help='finest processing resolution')
general.add_argument('-cs', '--crop_scale', default=1.2, type=float, metavar='F',
                     help='crop scale relative to bounding box')
general.add_argument('--gpus', default=[0], nargs='+', type=int, metavar='N',
                     help='list of gpu ids to use')
general.add_argument('--cpu_only', action='store_true',
                     help='force cpu only')
general.add_argument('-d', '--display', action='store_true',
                     help='display the rendering')
general.add_argument('-v', '--verbose', default=0, type=int, metavar='N',
                     help='verbose level')
general.add_argument('-ec', '--encoder_codec', default='avc1', metavar='STR',
                     help='encoder codec code')

detection = base_parser.add_argument_group('detection')
detection.add_argument('-dm', '--detection_model', metavar='PATH',
                       default=base_path + '/weights/WIDERFace_DSFD_RES152.pth',
                       help='path to face detection model')
detection.add_argument('-db', '--det_batch_size', default=det_batch_size, type=int, metavar='N',
                       help='detection batch size')
detection.add_argument('-dp', '--det_postfix', default='_dsfd.pkl', metavar='POSTFIX',
                       help='detection file postfix')

sequences = base_parser.add_argument_group('sequences')
sequences.add_argument('-it', '--iou_thresh', default=0.75, type=float,
                       metavar='F', help='IOU threshold')
sequences.add_argument('-ml', '--min_length', default=10, type=int,
                       metavar='N', help='minimum sequence length')
sequences.add_argument('-ms', '--min_size', default=64, type=int,
                       metavar='N', help='minimum sequence average bounding box size')
sequences.add_argument('-ck', '--center_kernel', default=25, type=int,
                       metavar='N', help='center average kernel size')
sequences.add_argument('-sk', '--size_kernel', default=51, type=int,
                       metavar='N', help='size average kernel size')
sequences.add_argument('-dsd', '--disable_smooth_det', dest='smooth_det', action='store_false',
                       help='disable smoothing the detection bounding boxes')
sequences.add_argument('-sp', '--seq_postfix', default='_dsfd_seq.pkl', metavar='POSTFIX',
                       help='sequence file postfix')
sequences.add_argument('-we', '--write_empty', action='store_true',
                       help='write empty sequence lists to file')

pose = base_parser.add_argument_group('pose')
pose.add_argument('-pm', '--pose_model', default=base_path + '/weights/hopenet_robust_alpha1.pth', metavar='PATH',
                  help='path to face pose model file')
pose.add_argument('-pb', '--pose_batch_size', default=128, type=int, metavar='N',
                  help='pose batch size')
pose.add_argument('-pp', '--pose_postfix', default='_pose.npz', metavar='POSTFIX',
                  help='pose file postfix')
pose.add_argument('-cp', '--cache_pose', action='store_true',
                  help='Toggle whether to cache pose')
pose.add_argument('-cf', '--cache_frontal', action='store_true',
                  help='Toggle whether to cache frontal images for each sequence')
pose.add_argument('-spo', '--smooth_poses', default=5, type=int, metavar='N',
                  help='poses temporal smoothing kernel size')

landmarks = base_parser.add_argument_group('landmarks')
landmarks.add_argument('-lm', '--lms_model', default=base_path + '/weights/hr18_wflw_landmarks.pth', metavar='PATH',
                       help='landmarks model')
landmarks.add_argument('-lb', '--lms_batch_size', default=64, type=int, metavar='N',
                       help='landmarks batch size')
landmarks.add_argument('-lp', '--landmarks_postfix', default='_lms.npz', metavar='POSTFIX',
                       help='landmarks file postfix')
landmarks.add_argument('-cl', '--cache_landmarks', action='store_true',
                       help='Toggle whether to cache landmarks')
landmarks.add_argument('-sl', '--smooth_landmarks', default=7, type=int, metavar='N',
                       help='landmarks temporal smoothing kernel size')

segmentation = base_parser.add_argument_group('segmentation')
segmentation.add_argument('-sm', '--seg_model', default=base_path + '/weights/celeba_unet_256_1_2_segmentation_v2.pth',
                          metavar='PATH', help='segmentation model')
segmentation.add_argument('-sb', '--seg_batch_size', default=32, type=int, metavar='N',
                          help='segmentation batch size')
segmentation.add_argument('-sep', '--segmentation_postfix', default='_seg.pkl', metavar='POSTFIX',
                          help='segmentation file postfix')
segmentation.add_argument('-cse', '--cache_segmentation', action='store_true',
                          help='Toggle whether to cache segmentation')
segmentation.add_argument('-sse', '--smooth_segmentation', default=5, type=int, metavar='N',
                          help='segmentation temporal smoothing kernel size')
segmentation.add_argument('-srm', '--seg_remove_mouth', action='store_true',
                          help='if true, the inner part of the mouth will be removed from the segmentation')

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 parents=[base_parser])
parser.add_argument('source', metavar='SOURCE', nargs='+',
                    help='image or video per source: files, directories, file lists or queries')
parser.add_argument('-t', '--target', metavar='TARGET', nargs='+',
                    help='video per target: files, directories, file lists or queries')
parser.add_argument('-o', '--output', metavar='DIR',
                    help='output directory')
parser.add_argument('-ss', '--select_source', default='longest', metavar='STR',
                    help='source selection method ["longest" | sequence number]')
parser.add_argument('-st', '--select_target', default='longest', metavar='STR',
                    help='target selection method ["longest" | sequence number]')
parser.add_argument('-b', '--batch_size', default=1, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('-rm', '--reenactment_model', metavar='PATH',
                    default=base_path + '/weights/nfv_msrunet_256_1_2_reenactment_v2.1.pth', help='reenactment model')
parser.add_argument('-cm', '--completion_model', default=base_path + '/weights/ijbc_msrunet_256_1_2_inpainting_v2.pth',
                    metavar='PATH', help='completion model')
parser.add_argument('-bm', '--blending_model', default=base_path + '/weights/ijbc_msrunet_256_1_2_blending_v2.pth',
                    metavar='PATH', help='blending model')
parser.add_argument('-ci', '--criterion_id',
                    default="vgg_loss.VGGLoss('" + base_path + "/weights/vggface2_vgg19_256_1_2_id.pth')",
                    metavar='OBJ', help='id criterion object')
parser.add_argument('-mr', '--min_radius', default=2.0, type=float, metavar='F',
                    help='minimum distance between points in the appearance map')
parser.add_argument('-oc', '--output_crop', action='store_true',
                    help='output crop around the face instead of full frame')
parser.add_argument('-rp', '--renderer_process', action='store_true',
                    help='If True, the renderer will be run in a separate process')
# preprocess
parser.add_argument('input', metavar='VIDEO', nargs='+',
                    help='path to input video')

# swap.py
finetune = parser.add_argument_group('finetune')
finetune.add_argument('-f', '--finetune', action='store_true',
                      help='Toggle whether to finetune the reenactment generator (default: False)')
finetune.add_argument('-fi', '--finetune_iterations', default=800, type=int, metavar='N',
                      help='number of finetune iterations')
finetune.add_argument('-fl', '--finetune_lr', default=1e-4, type=float, metavar='F',
                      help='finetune learning rate')
finetune.add_argument('-fb', '--finetune_batch_size', default=finetune_batch_size, type=int, metavar='N',
                      help='finetune batch size')
finetune.add_argument('-fw', '--finetune_workers', default=4, type=int, metavar='N',
                      help='finetune workers')
finetune.add_argument('-fs', '--finetune_save', action='store_true',
                      help='enable saving finetune checkpoint')

d = parser.get_default


class FaceDetectorImage(object):
    def __init__(self, out_postfix='_dsfd.pkl', detection_model_path=d('detection_model'),
                 device=None, gpus=None, batch_size=8, verbose=0):
        super(FaceDetectorImage, self).__init__()

        self.out_postfix = out_postfix
        self.batch_size = batch_size
        self.verbose = verbose

        # Set default tensor type

        # Initialize device

        if device is None or gpus is None:
            self.device, self.gpus = set_device(gpus)
        else:
            self.device, self.gpus = device, gpus

        # Initialize detection model
        self.net = SSD("test").to(self.device)
        self.net.load_state_dict(torch.load(detection_model_path))
        self.net.eval()

        # Initialize configuration
        self.transform = TestBaseTransform((104, 117, 123))
        self.cfg = widerface_640
        self.thresh = self.cfg['conf_thresh']

        # Support multiple GPUs
        if self.gpus and len(self.gpus) > 1:
            self.net = nn.DataParallel(self.net, self.gpus)

        self.net.requires_grad_(False)

        if HALF:
            self.net.half()

        # Reset default tensor type

    def __call__(self, input_image):

        frame_tensor = torch.from_numpy(self.transform(input_image)[0]).permute(2, 0, 1).unsqueeze(0)

        frame_tensor = frame_tensor.to(self.device)

        if HALF:
            frame_tensor = frame_tensor.half()

        # Process
        detections = self.net(frame_tensor)

        image_size = input_image.shape[:2]
        scale = torch.Tensor([image_size[1], image_size[0],
                              image_size[1], image_size[0]]).cpu().numpy()

        det = []
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= self.thresh:
                curr_det = detections[0, i, j, [1, 2, 3, 4, 0]].cpu().numpy()
                curr_det[:4] *= scale
                det.append(curr_det)
                j += 1

        # del detections

        if len(det) != 0:
            det = np.row_stack((det))
            det_filtered = det[det[:, 4] > 0.5, :4]
            return det_filtered
        return []


def crop_image_sequences_main_return_one(input_image, det, resolution=256, crop_scale=1.2):
    # img = np.copy(input_image)
    img = input_image

    # Crop image
    bbox = np.concatenate((det[:2], det[2:] - det[:2]))
    bbox = scale_bbox(bbox, crop_scale)
    img_cropped = crop_img(img, bbox)
    img_cropped = cv2.resize(img_cropped, (resolution, resolution), interpolation=cv2.INTER_CUBIC)

    return [img_cropped, bbox]


class AppearanceMapData(data.Dataset):
    """A dataset representing the appearance map of a video sequence

    Args:
        root (string): Root directory path or file list path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
     Attributes:
        video_paths (list): List of video paths
    """

    def __init__(self, src_transform=None, tgt_transform=None):
        self.src_transform = src_transform
        self.tgt_transform = tgt_transform
        self.src_vid = None

    def set_src(self, src_vid_seq_path,
                landmarks_postfix='_lms.npz', pose_postfix='_pose.npz', seg_postfix='_seg.pkl', min_radius=0.5):

        self.src_vid_seq_path = src_vid_seq_path
        self.src_vid = cv2.VideoCapture(self.src_vid_seq_path)

        src_lms_path = os.path.splitext(src_vid_seq_path)[0] + landmarks_postfix
        self.src_landmarks = np.load(src_lms_path)['landmarks_smoothed']
        src_pose_path = os.path.splitext(src_vid_seq_path)[0] + pose_postfix
        self.src_poses = np.load(src_pose_path)['poses_smoothed']

        # Initialize appearance map
        self.filtered_indices = fuse_clusters(self.src_poses[:, :2], r=min_radius / 99.)

        self.points = self.src_poses[self.filtered_indices, :2]
        limit_points = np.array([[-75., -75.], [-75., 75.], [75., -75.], [75., 75.]]) / 99.
        self.points = np.concatenate((self.points, limit_points))
        self.tri = Delaunay(self.points)
        self.valid_size = len(self.filtered_indices)

        # Filter source landmarks and poses and handle edge cases
        self.src_landmarks = self.src_landmarks[self.filtered_indices]
        self.src_landmarks = np.vstack((self.src_landmarks, np.zeros_like(self.src_landmarks[-1:])))
        self.src_poses = self.src_poses[self.filtered_indices]
        self.src_poses = np.vstack((self.src_poses, np.zeros_like(self.src_poses[-1:])))

        # Initialize cached frames
        self.src_frames = [None for i in range(len(self.filtered_indices) + 1)]

        # Handle edge cases
        black_rgb = np.zeros((d('resolution'), d('resolution'), 3), dtype='uint8')
        self.src_frames[-1] = black_rgb

    def get_item(self, tgt_frame, tgt_pose):

        # print(tgt_pose)

        # Query source frames and meta-data given the current target pose
        query_point, tilt_angle = tgt_pose[:2], tgt_pose[2]
        tri_index = self.tri.find_simplex(query_point[:2])
        tri_vertices = self.tri.simplices[tri_index]
        tri_vertices = np.minimum(tri_vertices, self.valid_size)

        # Compute barycentric weights
        b = self.tri.transform[tri_index, :2].dot(query_point[:2] - self.tri.transform[tri_index, 2])
        bw = np.array([b[0], b[1], 1 - b.sum()], dtype=float_type)
        bw[tri_vertices >= self.valid_size] = 0.  # Set zero weight for edge points
        bw /= bw.sum()

        # Cache source frames
        for tv in np.sort(tri_vertices):
            if self.src_frames[tv] is None:
                self.src_vid.set(cv2.CAP_PROP_POS_FRAMES, self.filtered_indices[tv])
                ret, frame_bgr = self.src_vid.read()
                assert frame_bgr is not None, 'Failed to read frame from source video in index: %d' % tv
                frame_rgb = frame_bgr[:, :, ::-1]
                self.src_frames[tv] = frame_rgb

        # Get source data from appearance map
        src_frames = [self.src_frames[tv] for tv in tri_vertices]
        src_landmarks = self.src_landmarks[tri_vertices].astype(float_type)
        src_poses = self.src_poses[tri_vertices].astype(float_type)

        # Apply source transformation
        if self.src_transform is not None:
            src_data = [(src_frames[i], src_landmarks[i], (src_poses[i][2] - tilt_angle) * 99.)
                        for i in range(len(src_frames))]
            src_data = self.src_transform(src_data)
            src_landmarks = torch.stack([src_data[i][1] for i in range(len(src_data))])
            src_frames = [src_data[i][0] for i in range(len(src_data))]
            src_poses[:, 2] = tilt_angle

        # Apply target transformation
        if self.tgt_transform is not None:
            tgt_frame = self.tgt_transform(tgt_frame)

        # Combine pyramids in source frames if they exist
        if isinstance(src_frames[0], (list, tuple)):
            src_frames = [torch.stack([src_frames[f][p] for f in range(len(src_frames))], dim=0)
                          for p in range(len(src_frames[0]))]

        return src_frames, src_landmarks, src_poses, bw, tgt_frame


class VideoProcessBaseFrameOneFace(object):
    def __init__(self, resolution=d('resolution'), crop_scale=d('crop_scale'), gpus=d('gpus'),
                 cpu_only=d('cpu_only'), display=d('display'), verbose=d('verbose'), encoder_codec=d('encoder_codec'),
                 # Detection arguments:
                 detection_model=d('detection_model'), det_batch_size=d('det_batch_size'), det_postfix=d('det_postfix'),
                 # Sequence arguments:
                 iou_thresh=d('iou_thresh'), min_length=d('min_length'), min_size=d('min_size'),
                 center_kernel=d('center_kernel'), size_kernel=d('size_kernel'), smooth_det=d('smooth_det'),
                 seq_postfix=d('seq_postfix'), write_empty=d('write_empty'),
                 # Pose arguments:
                 pose_model=d('pose_model'), pose_batch_size=d('pose_batch_size'), pose_postfix=d('pose_postfix'),
                 cache_pose=d('cache_pose'), cache_frontal=d('cache_frontal'), smooth_poses=d('smooth_poses'),
                 # Landmarks arguments:
                 lms_model=d('lms_model'), lms_batch_size=d('lms_batch_size'), landmarks_postfix=d('landmarks_postfix'),
                 cache_landmarks=d('cache_landmarks'), smooth_landmarks=d('smooth_landmarks'),
                 # Segmentation arguments:
                 seg_model=d('seg_model'), seg_batch_size=d('seg_batch_size'),
                 segmentation_postfix=d('segmentation_postfix'),
                 cache_segmentation=d('cache_segmentation'), smooth_segmentation=d('smooth_segmentation'),
                 seg_remove_mouth=d('seg_remove_mouth')):

        # Initialize device
        torch.set_grad_enabled(False)
        self.device, self.gpus = set_device(gpus, not cpu_only)

        # General
        self.resolution = resolution
        self.crop_scale = crop_scale
        self.display = display
        self.verbose = verbose

        # Detection
        self.face_detector = FaceDetectorImage(device=self.device, gpus=self.gpus)

        # FaceDetectorImage(det_postfix, detection_model, gpus, det_batch_size, display)
        self.det_postfix = det_postfix

        # Sequences
        self.iou_thresh = iou_thresh
        self.min_length = min_length
        self.min_size = min_size
        self.center_kernel = center_kernel
        self.size_kernel = size_kernel
        self.smooth_det = smooth_det
        self.seq_postfix = seq_postfix
        self.write_empty = write_empty

        # Pose
        self.pose_batch_size = pose_batch_size
        self.pose_postfix = pose_postfix
        self.cache_pose = cache_pose
        self.cache_frontal = cache_frontal
        self.smooth_poses = smooth_poses

        # Landmarks
        self.smooth_landmarks = smooth_landmarks
        self.landmarks_postfix = landmarks_postfix
        self.cache_landmarks = cache_landmarks
        self.lms_batch_size = lms_batch_size

        # Segmentation
        self.smooth_segmentation = smooth_segmentation
        self.segmentation_postfix = segmentation_postfix
        self.cache_segmentation = cache_segmentation
        self.seg_batch_size = seg_batch_size
        self.seg_remove_mouth = seg_remove_mouth and cache_landmarks

        # Load models
        self.face_pose = load_model(pose_model, 'face pose', self.device) if cache_pose else None
        self.L = load_model(lms_model, 'face landmarks', self.device) if cache_landmarks else None
        self.S = load_model(seg_model, 'face segmentation', self.device) if cache_segmentation else None

        # Initialize heatmap encoder
        self.heatmap_encoder = LandmarksHeatMapEncoder().to(self.device)

        # Initialize normalization tensors
        # Note: this is necessary because of the landmarks model
        self.img_mean = torch.as_tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        self.img_std = torch.as_tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        self.context_mean = torch.as_tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.context_std = torch.as_tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        # Support multiple GPUs
        if self.gpus and len(self.gpus) > 1:
            self.face_pose = nn.DataParallel(self.face_pose, self.gpus) if self.face_pose is not None else None
            self.L = nn.DataParallel(self.L, self.gpus) if self.L is not None else None
            self.S = nn.DataParallel(self.S, self.gpus) if self.S is not None else None

        self.face_pose.requires_grad_(False)
        self.L.requires_grad_(False)
        self.S.requires_grad_(False)

        if HALF:
            self.face_pose.half()
            self.L.half()
            self.S.half()

        # Initialize temportal smoothing
        if smooth_segmentation > 0:
            self.smooth_seg = TemporalSmoothing(3, smooth_segmentation).to(self.device)
        else:
            self.smooth_seg = None

        # Initialize output videos format
        # self.encoder_codec = encoder_codec
        # self.fourcc = cv2.VideoWriter_fourcc(*encoder_codec)

        # Initialize transforms
        self.transform_pose = img_landmarks_transforms.Compose([
            Resize(224), ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        self.transform_landmarks = img_landmarks_transforms.Compose([
            ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.transform_segmentation = img_landmarks_transforms.Compose([
            ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def process_pose(self, input_image):

        # For each batch of frames in the input video
        # seq_poses = []
        frame_rgb = self.transform_pose(input_image)

        frame = frame_rgb.unsqueeze(0).to(self.device)
        poses = self.face_pose(frame).div_(99.)  # Yaw, Pitch, Roll
        seq_poses = poses.cpu().numpy()
        # seq_poses.append(poses.cpu().numpy())
        # seq_poses = np.concatenate(seq_poses)

        seq_poses_smoothed = smooth_poses(seq_poses, self.smooth_poses)

        return [seq_poses, seq_poses_smoothed]

    def process_landmarks(self, input_image):

        frame_rgb = self.transform_landmarks(input_image)

        frame = frame_rgb.unsqueeze(0).to(self.device)

        # For each batch of frames in the input video
        # seq_landmarks = []

        H = self.L(frame)
        landmarks = self.heatmap_encoder(H)
        seq_landmarks = landmarks.cpu().numpy()
        # seq_landmarks.append(landmarks.cpu().numpy())
        # seq_landmarks = np.concatenate(seq_landmarks)

        seq_landmarks_smoothed = smooth_landmarks_98pts(seq_landmarks, self.smooth_landmarks)

        return [seq_landmarks, seq_landmarks_smoothed]

    def process_segmentation(self, input_image, landmarks):

        frame_rgb = self.transform_segmentation(input_image)
        frame = frame_rgb.unsqueeze(0).to(self.device)

        # Note : r = 2
        r = self.smooth_seg.kernel_radius
        pad_prev, pad_next = r, r  # This initialization is only relevant if there is a leftover from last batch

        # Compute segmentation
        segmentation = self.S(frame)

        if segmentation.shape[0] > r:
            pad_prev, pad_next = r, min(r, self.seg_batch_size - frame.shape[0])
            segmentation = self.smooth_seg(segmentation, pad_prev=pad_prev, pad_next=pad_next)

        mask = segmentation.argmax(1) == 1

        curr_mask = mask[0].cpu().numpy()
        # if self.seg_remove_mouth:
        #    curr_mask = remove_inner_mouth(curr_mask, landmarks[1][frame_count])

        return curr_mask

    def cache(self, input_image, result_fd):

        result_crop, bbox = crop_image_sequences_main_return_one(input_image, result_fd)

        result_pose = self.process_pose(result_crop)
        result_landmarks = self.process_landmarks(result_crop)
        result_seg = self.process_segmentation(result_crop, result_landmarks)

        return result_crop, bbox, result_pose, result_landmarks, result_seg

    def detect(self, input_image):

        return self.face_detector(input_image)



class FaceSwappingFrame(VideoProcessBaseFrameOneFace):
    def __init__(self, resolution=d('resolution'), crop_scale=d('crop_scale'), gpus=d('gpus'),
                 cpu_only=d('cpu_only'), display=d('display'), verbose=d('verbose'), encoder_codec=d('encoder_codec'),
                 # Detection arguments:
                 detection_model=d('detection_model'), det_batch_size=d('det_batch_size'), det_postfix=d('det_postfix'),
                 # Sequence arguments:
                 iou_thresh=d('iou_thresh'), min_length=d('min_length'), min_size=d('min_size'),
                 center_kernel=d('center_kernel'), size_kernel=d('size_kernel'), smooth_det=d('smooth_det'),
                 seq_postfix=d('seq_postfix'), write_empty=d('write_empty'),
                 # Pose arguments:
                 pose_model=d('pose_model'), pose_batch_size=d('pose_batch_size'), pose_postfix=d('pose_postfix'),
                 cache_pose=d('cache_pose'), cache_frontal=d('cache_frontal'), smooth_poses=d('smooth_poses'),
                 # Landmarks arguments:
                 lms_model=d('lms_model'), lms_batch_size=d('lms_batch_size'), landmarks_postfix=d('landmarks_postfix'),
                 cache_landmarks=d('cache_landmarks'), smooth_landmarks=d('smooth_landmarks'),
                 # Segmentation arguments:
                 seg_model=d('seg_model'), smooth_segmentation=d('smooth_segmentation'),
                 segmentation_postfix=d('segmentation_postfix'), cache_segmentation=d('cache_segmentation'),
                 seg_batch_size=d('seg_batch_size'), seg_remove_mouth=d('seg_remove_mouth'),
                 # Finetune arguments:
                 finetune=d('finetune'), finetune_iterations=d('finetune_iterations'), finetune_lr=d('finetune_lr'),
                 finetune_batch_size=d('finetune_batch_size'), finetune_workers=d('finetune_workers'),
                 finetune_save=d('finetune_save'),
                 # Swapping arguments:
                 batch_size=d('batch_size'), reenactment_model=d('reenactment_model'),
                 completion_model=d('completion_model'),
                 blending_model=d('blending_model'), criterion_id=d('criterion_id'), min_radius=d('min_radius'),
                 output_crop=d('output_crop'), renderer_process=d('renderer_process')):

        super(FaceSwappingFrame, self).__init__(
            resolution, crop_scale, gpus, cpu_only, display, verbose, encoder_codec,
            detection_model=detection_model, det_batch_size=det_batch_size, det_postfix=det_postfix,
            iou_thresh=iou_thresh, min_length=min_length, min_size=min_size, center_kernel=center_kernel,
            size_kernel=size_kernel, smooth_det=smooth_det, seq_postfix=seq_postfix, write_empty=write_empty,
            pose_model=pose_model, pose_batch_size=pose_batch_size, pose_postfix=pose_postfix,
            cache_pose=True, cache_frontal=cache_frontal, smooth_poses=smooth_poses,
            lms_model=lms_model, lms_batch_size=lms_batch_size, landmarks_postfix=landmarks_postfix,
            cache_landmarks=True, smooth_landmarks=smooth_landmarks, seg_model=seg_model,
            seg_batch_size=seg_batch_size, segmentation_postfix=segmentation_postfix,
            cache_segmentation=True, smooth_segmentation=smooth_segmentation, seg_remove_mouth=seg_remove_mouth)

        self.batch_size = batch_size
        self.min_radius = min_radius
        self.output_crop = output_crop
        self.finetune_enabled = finetune
        self.finetune_iterations = finetune_iterations
        self.finetune_lr = finetune_lr
        self.finetune_batch_size = finetune_batch_size
        self.finetune_workers = finetune_workers
        self.finetune_save = finetune_save

        self.reenactment_model = reenactment_model

        # Load reenactment model
        self.Gr, checkpoint = load_model(reenactment_model, 'face reenactment', self.device, return_checkpoint=True)
        self.Gr.arch = checkpoint['arch']
        self.reenactment_state_dict = checkpoint['state_dict']

        # Load all other models
        self.Gc = load_model(completion_model, 'face completion', self.device)
        self.Gb = load_model(blending_model, 'face blending', self.device)

        # Initialize landmarks decoders
        self.landmarks_decoders = []
        for res in (128, 256):
            self.landmarks_decoders.insert(0, LandmarksHeatMapDecoder(res).to(self.device))

        # Initialize losses
        self.criterion_pixelwise = nn.L1Loss().to(self.device)
        self.criterion_id = obj_factory(criterion_id).to(self.device)

        # Support multiple GPUs
        if self.gpus and len(self.gpus) > 1:
            self.Gr = nn.DataParallel(self.Gr, self.gpus)
            self.Gc = nn.DataParallel(self.Gc, self.gpus)
            self.Gb = nn.DataParallel(self.Gb, self.gpus)
            self.criterion_id.vgg = nn.DataParallel(self.criterion_id.vgg, self.gpus)

        self.Gr.requires_grad_(False)
        self.Gc.requires_grad_(False)
        self.Gb.requires_grad_(False)
        self.criterion_id.vgg.requires_grad_(False)

        if HALF:
            self.Gr.half()
            self.Gc.half()
            self.Gb.half()
            self.criterion_id.vgg.half()

        # Initialize soft erosion
        src_transform = img_lms_pose_transforms.Compose([Rotate(), Pyramids(2), ToTensor(), Normalize()])
        tgt_transform = img_lms_pose_transforms.Compose([ToTensor(), Normalize()])
        self.smooth_mask = SoftErosion(kernel_size=21, threshold=0.6).to(self.device)
        self.appearance_map = AppearanceMapData(src_transform, tgt_transform)

        # torch.set_default_tensor_type(torch.FloatTensor)
        # if HALF:
        #    torch.set_default_tensor_type(torch.HalfTensor)

    def load(self, input_path):
        output_dir = os.path.splitext(input_path)[0]
        seq_file_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input_path))[0] + self.seq_postfix)
        pose_file_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input_path))[0] + self.pose_postfix)

        return output_dir, seq_file_path, pose_file_path

    def finetune(self, source_path):
        checkpoint_path = os.path.splitext(source_path)[0] + '_Gr.pth'
        print(checkpoint_path)
        if os.path.isfile(checkpoint_path):
            print('=> Loading the reenactment generator finetuned on: "%s"...' % os.path.basename(source_path))
            checkpoint = torch.load(checkpoint_path)
            if self.gpus and len(self.gpus) > 1:
                self.Gr.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.Gr.load_state_dict(checkpoint['state_dict'])
            return
        else:
            print('=> err: no finetuned model')
            self.Gr, checkpoint = load_model(self.reenactment_model, 'face reenactment', self.device,
                                             return_checkpoint=True)

    def prepare(self, source_path, select_source='longest'):
        source_cache_dir, source_seq_file_path, _ = self.load(source_path)

        self.source_cache_dir = source_cache_dir
        self.source_seq_file_path = source_seq_file_path

        with open(source_seq_file_path, "rb") as fp:  # Unpickling
            source_seq_list = pickle.load(fp)

        self.source_seq = select_seq(source_seq_list, select_source)

        src_path_no_ext, src_ext = os.path.splitext(source_path)
        src_vid_seq_name = os.path.basename(src_path_no_ext) + '_seq%02d%s' % (self.source_seq.id, src_ext)
        src_vid_seq_path = os.path.join(source_cache_dir, src_vid_seq_name)

        self.finetune(src_vid_seq_path)

        self.appearance_map.set_src(src_vid_seq_path, min_radius=self.min_radius)

    def __call__(self, tgt_image, select='biggest'):
        target_fd_list = self.detect(tgt_image)

        # print('target_fd_list')
        # print(target_fd_list)

        indexes = []

        if len(target_fd_list) == 0:
            return np.array(tgt_image[:, :, ::-1])

        if select == 'all':
            indexes = range(len(target_fd_list))
        elif select.isnumeric():
            indexes.append(int(select))
        else:
            index = 0
            size = 0
            for i, target_fd in enumerate(target_fd_list):
                # x1, y1, x2, y2
                # target_seq = target_seq[0]
                now_size = (target_fd[0] - target_fd[2]) ** 2 + (target_fd[1] - target_fd[3]) ** 2

                if size > now_size:
                    size = now_size
                    index = i

            indexes.append(index)

        result_image = tgt_image[:, :, ::-1]

        # for each faces
        for index in indexes:
            target_fd = target_fd_list[index]

            tgt_crop_image, tgt_bbox, tgt_poses, tgt_landmarks, tgt_seg = self.cache(result_image, target_fd)

            tgt_crop_image = tgt_crop_image
            tgt_landmarks = tgt_landmarks[1]
            tgt_poses = tgt_poses[1][0]
            tgt_mask = tgt_seg

            # in original, it was for
            src_frame, src_landmarks, src_poses, bw, tgt_frame = self.appearance_map.get_item(tgt_crop_image, tgt_poses)

            tgt_landmarks = torch.from_numpy(tgt_landmarks)
            tgt_mask = torch.from_numpy(tgt_mask)

            bw = torch.from_numpy(np.array([bw]))

            # Prepare input
            for p in range(len(src_frame)):
                src_frame[p] = src_frame[p].unsqueeze(0).to(self.device)
            tgt_frame = tgt_frame.to(self.device)
            tgt_landmarks = tgt_landmarks.to(self.device)
            # tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0).int().to(self.device).bool()
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0).bool().to(self.device)

            bw = bw.to(self.device)
            bw_indices = torch.nonzero(torch.any(bw > 0, dim=0), as_tuple=True)[0]
            bw = bw[:, bw_indices]

            # For each source frame perform reenactment
            reenactment_triplet = []
            for j in bw_indices:
                input = []
                for p in range(len(src_frame)):
                    context = self.landmarks_decoders[p](tgt_landmarks)

                    input.append(torch.cat((src_frame[p][:, j], context), dim=1))

                # Reenactment
                reenactment_triplet.append(self.Gr(input).unsqueeze(1))
            reenactment_tensor = torch.cat(reenactment_triplet, dim=1)

            # Barycentric interpolation of reenacted frames
            reenactment_tensor = (reenactment_tensor * bw.view(*bw.shape, 1, 1, 1)).sum(dim=1)

            # Compute reenactment segmentation
            reenactment_seg = self.S(reenactment_tensor)
            reenactment_background_mask_tensor = (reenactment_seg.argmax(1) != 1).unsqueeze(1)

            # Remove the background of the aligned face
            reenactment_tensor.masked_fill_(reenactment_background_mask_tensor, -1.0)

            # Soften target mask
            soft_tgt_mask, eroded_tgt_mask = self.smooth_mask(tgt_mask)

            # Complete face
            inpainting_input_tensor = torch.cat((reenactment_tensor, eroded_tgt_mask.float()), dim=1)
            inpainting_input_tensor_pyd = create_pyramid(inpainting_input_tensor, 2)
            completion_tensor = self.Gc(inpainting_input_tensor_pyd)

            tgt_frame = tgt_frame.unsqueeze(0)

            # Blend faces
            transfer_tensor = transfer_mask(completion_tensor, tgt_frame, eroded_tgt_mask)
            blend_input_tensor = torch.cat((transfer_tensor, tgt_frame, eroded_tgt_mask.float()), dim=1)
            blend_input_tensor_pyd = create_pyramid(blend_input_tensor, 2)
            blend_tensor = self.Gb(blend_input_tensor_pyd)

            result_tensor = blend_tensor * soft_tgt_mask + tgt_frame * (1 - soft_tgt_mask)  # bgr image

            result = result_tensor.cpu()[0]
            result = np.transpose(result, axes=(1, 2, 0)).numpy()
            result = (result - np.min(result)) / np.ptp(result) * 255

            result_image = crop2img(result_image, result, tgt_bbox)

        return result_image


model = FaceSwappingFrame()
# 여기까지 전처리

p = base_path + '/docs/examples/'
source = p + 'V.mp4'
# 모델 교체
model.prepare(source)

# GUI 표시
# test
# from luxus_test.test_pyqt import test

# test(model)