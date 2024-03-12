import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
import torchvision.transforms as transforms
from cog import BasePredictor, Path, Input
from demo import load_checkpoints
from demo import make_animation
from demo import make_animation_by_video
from demo import align_image

class Predictor(BasePredictor):
    def setup(self):

        self.device = torch.device("cuda:0")
        datasets = ["vox", "taichi", "ted", "mgif"]
        (
            self.inpainting,
            self.kp_detector,
            self.dense_motion_network,
            self.avd_network,
        ) = ({}, {}, {}, {})
        for d in datasets:
            (
                self.inpainting[d],
                self.kp_detector[d],
                self.dense_motion_network[d],
                self.avd_network[d],
            ) = load_checkpoints(
                config_path=f"config/{d}-384.yaml"
                if d == "ted"
                else f"config/{d}-256.yaml",
                checkpoint_path=f"checkpoints/{d}.pth.tar",
                device=self.device,
            )

    def predict(
        self,
        source_image: Path = Input(
            description="Input source image.",
        ),
        driving_video: Path = Input(
            description="Choose a micromotion.",
        ),
        dataset_name: str = Input(
            choices=["vox", "taichi", "ted", "mgif"],
            default="vox",
            description="Choose a dataset.",
        ),
        result_name: str = "output.mp4",
        find_best_frame: bool = False
    ) -> Path:

        predict_mode = "relative"  # ['standard', 'relative', 'avd']
        # find_best_frame = False

        pixel = 384 if dataset_name == "ted" else 256

        if dataset_name == "vox":
            # first run face alignment
            align_image(str(source_image), 'aligned.png')
            source_image = imageio.imread('aligned.png')
        else:
            source_image = imageio.imread(str(source_image))
        reader = imageio.get_reader(str(driving_video))
        fps = reader.get_meta_data()["fps"]
        source_image = resize(source_image, (pixel, pixel))[..., :3]

        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()
        
        driving_video = [
            resize(frame, (pixel, pixel))[..., :3] for frame in driving_video
        ]

        inpainting, kp_detector, dense_motion_network, avd_network = (
            self.inpainting[dataset_name],
            self.kp_detector[dataset_name],
            self.dense_motion_network[dataset_name],
            self.avd_network[dataset_name],
        )
        
        
        if predict_mode=='relative' and find_best_frame:
            from demo import find_best_frame as _find
            i = _find(source_image, driving_video, False)
            print ("Best frame: " + str(i))
            driving_forward = driving_video[i:]
            driving_backward = driving_video[:(i+1)][::-1]
            predictions_forward = make_animation(
                source_image,
                driving_forward,
                inpainting,
                kp_detector,
                dense_motion_network,
                avd_network,
                device="cuda:0",
                mode=predict_mode,
            )
            predictions_backward = make_animation(
                source_image,
                driving_backward,
                inpainting,
                kp_detector,
                dense_motion_network,
                avd_network,
                device="cuda:0",
                mode=predict_mode,
            )
            predictions = predictions_backward[::-1] + predictions_forward[1:]
        else:
            predictions = make_animation(
                source_image,
                driving_video,
                inpainting,
                kp_detector,
                dense_motion_network,
                avd_network,
                device="cuda:0",
                mode=predict_mode,
            )

        # save resulting video
        out_path = Path("./results") / result_name
        imageio.mimsave(
            str(out_path), [img_as_ubyte(frame) for frame in predictions], fps=fps
        )
        return out_path

    def predict_by_video(
        self,
        source_video: Path = Input(
            description="Input source video.",
        ),
        driving_video: Path = Input(
            description="Choose a micromotion.",
        ),
        dataset_name: str = Input(
            choices=["vox", "taichi", "ted", "mgif"],
            default="vox",
            description="Choose a dataset.",
        ),
        result_name: str = "output.mp4",
    ) -> Path:

        predict_mode = "relative"  # ['standard', 'relative', 'avd']
        # find_best_frame = False

        pixel = 384 if dataset_name == "ted" else 256

        # if dataset_name == "vox":
        #     # first run face alignment
        #     align_image(str(source_image), 'aligned.png')
        #     source_image = imageio.imread('aligned.png')
        # else:
        #     source_image = imageio.imread(str(source_image))
        
        reader_source = imageio.get_reader(str(source_video))
        reader = imageio.get_reader(str(driving_video))
        fps = reader.get_meta_data()["fps"]
        # source_image = resize(source_image, (pixel, pixel))[..., :3]
        
        # read source video and driving video
        source_video = []
        try:
            for im in reader_source:
                source_video.append(im)
        except RuntimeError:
            pass
        reader_source.close()
        
        source_video = [
            resize(frame, (pixel, pixel))[..., :3] for frame in source_video
        ]
        
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()

        driving_video = [
            resize(frame, (pixel, pixel))[..., :3] for frame in driving_video
        ]

        inpainting, kp_detector, dense_motion_network, avd_network = (
            self.inpainting[dataset_name],
            self.kp_detector[dataset_name],
            self.dense_motion_network[dataset_name],
            self.avd_network[dataset_name],
        )
        
        predictions = make_animation_by_video(
            source_video,
            driving_video,
            inpainting,
            kp_detector,
            dense_motion_network,
            avd_network,
            device="cuda:0",
            mode=predict_mode,
            dataset_name=dataset_name
        )
        
        # save resulting video
        out_path = Path("./results") / result_name
        imageio.mimsave(
            str(out_path), [img_as_ubyte(frame) for frame in predictions], fps=fps
        )
        return out_path


import time
import pandas as pd

if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()
    source_image_path_list = [Path("./assets/source.png"),
                              Path("./assets/source_glass.png"), 
                              Path("./assets/source_noglass.png"),
                              Path("./assets/source_ai.png"),
                              Path("./assets/source_3D.png")]
    
    driving_video_path_list = [Path("./assets/driving.mp4"),
                               Path("./assets/driving3D.mp4"),
                               Path("./assets/driving3D_all.mp4"),
                               Path("./assets/driving3d_fs.mp4"),
                               Path("./assets/driving3d_fl.mp4"),]

    dataset_name = "vox"  # or "taichi", "ted", "mgif"

    # Create an empty DataFrame to store the results
    results = pd.DataFrame(columns=["source_image", "driving_video", "find_best_frame", "result_path", "elapsed_time"])
    
    start_time = time.time()
    source_image_path = source_image_path_list[2]
    source_video_path = driving_video_path_list[0]
    driving_video_path = driving_video_path_list[3]
    result_name = f"output_demo_video.mp4"
    # result_path = predictor.predict(source_image_path, driving_video_path, dataset_name, result_name, find_best_frame=False)
    result_path = predictor.predict_by_video(source_video_path, driving_video_path, dataset_name, result_name)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The {result_path} took {elapsed_time} seconds to run.")

    # for i in range(4):
    #     for j in range(5):
    #         for k in (True, False):
    #             start_time = time.time()
    #             source_image_path = source_image_path_list[i]
    #             driving_video_path = driving_video_path_list[j]
    #             result_name = f"output_{i}_{j}_{k}.mp4"
    #             result_path = predictor.predict(source_image_path, driving_video_path, dataset_name, result_name, find_best_frame=k)
    #             print(f"Result video is saved at {result_path}")
    #             end_time = time.time()
    #             elapsed_time = end_time - start_time
    #             print(f"The {result_path} took {elapsed_time} seconds to run.")

    #             # Add the result to the DataFrame
    #             results = results.append({
    #                 "source_image": source_image_path,
    #                 "driving_video": driving_video_path,
    #                 "find_best_frame": k,
    #                 "result_path": result_path,
    #                 "elapsed_time": elapsed_time
    #             }, ignore_index=True)

    # # Save the DataFrame to a CSV file
    # results.to_csv("./results/results.csv", index=False)

