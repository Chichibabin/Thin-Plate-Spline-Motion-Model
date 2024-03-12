import matplotlib
matplotlib.use('Agg')
import sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
from scipy.spatial import ConvexHull
import numpy as np
import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from modules.inpainting_network import InpaintingNetwork
from modules.keypoint_detector import KPDetector
from modules.dense_motion import DenseMotionNetwork
from modules.avd_network import AVDNetwork
import PIL.Image
import os
import sys
import scipy
import dlib
import warnings

def image_align(src_file, dst_file, face_landmarks, output_size=1024, transform_size=4096, enable_padding=True):
        # Align function from FFHQ dataset pre-processing step
        # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

        lm = np.array(face_landmarks)
        lm_chin          = lm[0  : 17]  # left-right
        lm_eyebrow_left  = lm[17 : 22]  # left-right
        lm_eyebrow_right = lm[22 : 27]  # left-right
        lm_nose          = lm[27 : 31]  # top-down
        lm_nostrils      = lm[31 : 36]  # top-down
        lm_eye_left      = lm[36 : 42]  # left-clockwise
        lm_eye_right     = lm[42 : 48]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60]  # left-clockwise
        lm_mouth_inner   = lm[60 : 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # Load in-the-wild image.
        if not os.path.isfile(src_file):
            print('\nCannot find source image. Please run "--wilds" before "--align".')
            return
        img = PIL.Image.open(src_file)

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.LANCZOS)

        # Save aligned image.
        img.save(dst_file, 'PNG')
     
class LandmarksDetector:
    def __init__(self, predictor_model_path):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        self.detector = dlib.get_frontal_face_detector() # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def get_landmarks(self, image):
        img = dlib.load_rgb_image(image)
        dets = self.detector(img, 1)

        for detection in dets:
            face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
            yield face_landmarks

warnings.filterwarnings("ignore")

PREDICTOR = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
LANDMARKS_DETECTOR = LandmarksDetector("shape_predictor_68_face_landmarks.dat")

def align_image(raw_img_path, aligned_face_path):
    for i, face_landmarks in enumerate(LANDMARKS_DETECTOR.get_landmarks(raw_img_path), start=1):
        image_align(raw_img_path, aligned_face_path, face_landmarks)

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.9")

def relative_kp(kp_source, kp_driving, kp_driving_initial):

    source_area = ConvexHull(kp_source['fg_kp'][0].data.cpu().numpy()).volume
    driving_area = ConvexHull(kp_driving_initial['fg_kp'][0].data.cpu().numpy()).volume
    adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)

    kp_new = {k: v for k, v in kp_driving.items()}

    kp_value_diff = (kp_driving['fg_kp'] - kp_driving_initial['fg_kp'])
    kp_value_diff *= adapt_movement_scale
    kp_new['fg_kp'] = kp_value_diff + kp_source['fg_kp']

    return kp_new

def load_checkpoints(config_path, checkpoint_path, device):
    with open(config_path) as f:
        config = yaml.full_load(f)

    inpainting = InpaintingNetwork(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    kp_detector = KPDetector(**config['model_params']['common_params'])
    dense_motion_network = DenseMotionNetwork(**config['model_params']['common_params'],
                                              **config['model_params']['dense_motion_params'])
    avd_network = AVDNetwork(num_tps=config['model_params']['common_params']['num_tps'],
                             **config['model_params']['avd_network_params'])
    kp_detector.to(device)
    dense_motion_network.to(device)
    inpainting.to(device)
    avd_network.to(device)
       
    checkpoint = torch.load(checkpoint_path, map_location=device)
 
    inpainting.load_state_dict(checkpoint['inpainting_network'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    dense_motion_network.load_state_dict(checkpoint['dense_motion_network'])
    if 'avd_network' in checkpoint:
        avd_network.load_state_dict(checkpoint['avd_network'])
    
    inpainting.eval()
    kp_detector.eval()
    dense_motion_network.eval()
    avd_network.eval()
    
    return inpainting, kp_detector, dense_motion_network, avd_network


def make_animation(source_image, driving_video, inpainting_network, kp_detector, dense_motion_network, avd_network, device, mode = 'relative'):
    assert mode in ['standard', 'relative', 'avd']
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        source = source.to(device)
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3).to(device)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            driving_frame = driving_frame.to(device)
            kp_driving = kp_detector(driving_frame)
            if mode == 'standard':
                kp_norm = kp_driving
            elif mode=='relative':
                kp_norm = relative_kp(kp_source=kp_source, kp_driving=kp_driving,
                                    kp_driving_initial=kp_driving_initial)
            elif mode == 'avd':
                kp_norm = avd_network(kp_source, kp_driving)
            dense_motion = dense_motion_network(source_image=source, kp_driving=kp_norm,
                                                    kp_source=kp_source, bg_param = None, 
                                                    dropout_flag = False)
            out = inpainting_network(source, dense_motion)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions

def make_animation_by_video(source_video, driving_video, inpainting_network, kp_detector, dense_motion_network, avd_network, device, mode = 'relative', dataset_name = 'vox'):
    pixel = 384 if dataset_name == "ted" else 256
    assert mode in ['standard', 'relative', 'avd']
    with torch.no_grad():
        predictions = []

        source = torch.tensor(np.array(source_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3).to(device)
        
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3).to(device)
        # kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            driving_frame = driving_frame.to(device)
            kp_driving = kp_detector(driving_frame)
            i = find_best_frame(driving_video[frame_idx], source_video, False)
            
            source_image = source_video[i]
            if dataset_name == "vox":
                imageio.imsave('tmp_frame.png', source_image)
                align_image('tmp_frame.png', 'aligned_frame.png')
                source_image = imageio.imread('aligned_frame.png')
                
            source_image = resize(source_image, (pixel, pixel))[..., :3]
            source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            source = source.to(device)
            kp_source = kp_detector(source)
            
            if mode == 'standard':
                kp_norm = kp_driving
            elif mode=='relative':
                kp_norm = relative_kp(kp_source=kp_source, kp_driving=kp_driving,
                                    kp_driving_initial=kp_driving_initial)
            elif mode == 'avd':
                kp_norm = avd_network(kp_source, kp_driving)
            
            dense_motion = dense_motion_network(source_image=source, kp_driving=kp_norm,
                                                    kp_source=kp_source, bg_param = None, 
                                                    dropout_flag = False)
            out = inpainting_network(source, dense_motion)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions

def find_best_frame(source, driving, cpu):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=True,
                                      device= 'cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        try:
            kp_driving = fa.get_landmarks(255 * image)[0]
            kp_driving = normalize_kp(kp_driving)
            new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
            if new_norm < norm:
                norm = new_norm
                frame_num = i
        except:
            pass
    return frame_num


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='checkpoints/vox.pth.tar', help="path to checkpoint to restore")

    parser.add_argument("--source_image", default='./assets/source.png', help="path to source image")
    parser.add_argument("--driving_video", default='./assets/driving.mp4', help="path to driving video")
    parser.add_argument("--result_video", default='./result.mp4', help="path to output")
    
    parser.add_argument("--img_shape", default="256,256", type=lambda x: list(map(int, x.split(','))),
                        help='Shape of image, that the model was trained on.')
    
    parser.add_argument("--mode", default='relative', choices=['standard', 'relative', 'avd'], help="Animate mode: ['standard', 'relative', 'avd'], when use the relative mode to animate a face, use '--find_best_frame' can get better quality result")
    
    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true", 
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")

    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

    opt = parser.parse_args()

    source_image = imageio.imread(opt.source_image)
    reader = imageio.get_reader(opt.driving_video)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()
    
    if opt.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    
    source_image = resize(source_image, opt.img_shape)[..., :3]
    driving_video = [resize(frame, opt.img_shape)[..., :3] for frame in driving_video]
    inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(config_path = opt.config, checkpoint_path = opt.checkpoint, device = device)
 
    if opt.find_best_frame:
        i = find_best_frame(source_image, driving_video, opt.cpu)
        print ("Best frame: " + str(i))
        driving_forward = driving_video[i:]
        driving_backward = driving_video[:(i+1)][::-1]
        predictions_forward = make_animation(source_image, driving_forward, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = opt.mode)
        predictions_backward = make_animation(source_image, driving_backward, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = opt.mode)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        predictions = make_animation(source_image, driving_video, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = opt.mode)
    
    imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)

