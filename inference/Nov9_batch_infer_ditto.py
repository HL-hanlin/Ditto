import argparse
import torch
import os
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

import pandas as pd
import argparse
import torch.distributed as dist
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List
import torch
from PIL import Image
import imageio

# from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video, load_video




# --------------------------
# 1. Parse arguments
# --------------------------
# parser = argparse.ArgumentParser()
# parser.add_argument("--start_idx", type=int, default=0, help="Start row index for slicing")
# parser.add_argument("--end_idx", type=int, default=None, help="End row index for slicing (exclusive)")
# args = parser.parse_args()


# --------------------------
# 2. Distributed setup
# --------------------------
dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ["LOCAL_RANK"])  # torchrun sets this

"""
rank = 0
world_size = 1
local_rank = 0
"""


device = torch.device(f"cuda:{local_rank}")
print(f"[Rank {rank}] starting on device {device}")



def main(args):

    # device = f"cuda:{args.device_id}"

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=str(device),
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),#,local_model_path="/fsx-project/hanlin/checkpoints/Wan2.1-VACE-14B"),
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),#,local_model_path="/fsx-project/hanlin/checkpoints/Wan2.1-VACE-14B"),
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),#,local_model_path="/fsx-project/hanlin/checkpoints/Wan2.1-VACE-14B"),
        ],
        # redirect_common_files=False
    )

    args.lora_path = '/fsx-project/hanlin/checkpoints/Ditto_models/ditto_global.safetensors'
    if args.lora_path:
        print(f"Loading Ditto LoRA model: {args.lora_path} (alpha={args.lora_alpha})")
        # if not os.path.exists(args.lora_path):
        #     print(f"Error: LoRA file not found at {args.lora_path}")
        #     return
        pipe.load_lora(pipe.vace, args.lora_path, alpha=args.lora_alpha)

    pipe.enable_vram_management()

    # print(f"Loading input video: {args.input_video}")
    # if not os.path.exists(args.input_video):
    #     print(f"Error: Input video file not found at {args.input_video}")
    #     return

    args.height = 480
    args.width = 832
    num_frames = 121




    # --------------------------
    # 2. Load dataset
    # --------------------------

    test_dataset = pd.read_csv("/fsx-project/hanlin/data/sstk_v2v_eval_3k/sstk_v2v_eval_304.csv")
    test_dataset['media_id'] = test_dataset['source_path'].apply(lambda x: x.split('/')[-1].replace(".mp4", ""))
    metadatas = []
    for i in range(len(test_dataset)):
        item = {
            'key': str(test_dataset.iloc[i]['media_id']),
            'task_type': test_dataset.iloc[i]['type'],
            'instruction': test_dataset.iloc[i]['instruction'],
            'input_images': test_dataset.iloc[i]['source_path'],
        }
        metadatas.append(item)
    print(f'Filtered dataset size: {len(metadatas)}')
    end_idx = args.end_idx if args.end_idx is not None else len(metadatas)
    metadatas = metadatas[args.start_idx:end_idx]


    # Split by rank
    num_rows = len(metadatas)
    rows_per_rank = (num_rows + world_size - 1) // world_size
    start = rank * rows_per_rank
    end = min(start + rows_per_rank, num_rows)

    metadatas = metadatas[start:end]
    print(f"[Rank {rank}] processing rows {start}:{end} out of {num_rows}")






    # --------------------------
    # 3. Inference
    # --------------------------

    for index, metadata in enumerate(metadatas):


        url = metadata['input_images']
        prompt = metadata['instruction']

        num_frames = 121
        width = 832
        height = 480

        # Load video
        def convert_video(video: List[Image.Image]) -> List[Image.Image]:
            video = load_video(url)[:num_frames]
            video = [video[i].resize((width, height)) for i in range(num_frames)]
            return video


        cond_video = load_video(url, convert_method=convert_video)




        # video = VideoData(url, height=args.height, width=args.width)

        # num_frames = min(args.num_frames, len(cond_video))
        # if num_frames != args.num_frames:
        #     print(f"Warning: Requested number of frames ({args.num_frames}) exceeds total video frames ({len(video)}). Using {num_frames} frames instead.")

        cond_video = [cond_video[i] for i in range(num_frames)]

        reference_image = None

        video = pipe(
            prompt=prompt,
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            vace_video=cond_video,
            vace_reference_image=reference_image,
            num_frames=num_frames,
            seed=42,
            tiled=True,
        )



        # resized_output = np.array([cv2.resize(frame, (1280, 704), interpolation=cv2.INTER_LINEAR) for frame in output])


        target_size = (1280, 704)  # (width, height)
        resized_images = [img.resize(target_size, resample=Image.BILINEAR) for img in video]
        outpath = os.path.join("/fsx-project/hanlin/outputs/V2V_baselines/ditto_832w_480h_121f/sstk_v2v_eval_304", metadata['task_type'], metadata['key']+".mp4")
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        # export_to_video(resized_output, outpath, fps=24)
        imageio.mimsave(outpath, resized_images[:num_frames], fps=24)



        target_size = (1280, 704)  # (width, height)
        resized_cond_images = [img.resize(target_size, resample=Image.BILINEAR) for img in cond_video]
        cond_outpath = outpath.replace(".mp4", "_cond.mp4")
        imageio.mimsave(cond_outpath, resized_cond_images[:num_frames], fps=24)






        # output_dir = os.path.dirname(args.output_video)
        # if output_dir:
        #     os.makedirs(output_dir, exist_ok=True)

        # save_video(video, args.output_video, fps=args.fps, quality=args.quality)








if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="InstructV2V Pipeline.")

    parser.add_argument("--input_video", type=str, required=False, help="Path to the input video file.")
    parser.add_argument("--output_video", type=str, required=False, help="Path to save the output video file.")
    parser.add_argument("--lora_path", type=str, default=None, help="Optional path to a LoRA model file (.safetensors).")
    parser.add_argument("--device_id", type=int, default=0, help="The ID of the CUDA device to use (e.g., 0, 1, 2).")
    parser.add_argument("--prompt", type=str, required=False, help="The positive prompt describing the target style.")
    parser.add_argument("--height", type=int, default=480, help="The height to use for video processing.")
    parser.add_argument("--width", type=int, default=832, help="The width to use for video processing.")
    parser.add_argument("--num_frames", type=int, default=121, help="The number of video frames to process.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducible results.")
    parser.add_argument("--start_idx", type=int, default=0, help="Start row index for slicing")
    parser.add_argument("--end_idx", type=int, default=None, help="End row index for slicing (exclusive)")

    parser.add_argument("--lora_alpha", type=float, default=1.0, help="The alpha (weight) value for the LoRA model.")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second (FPS) for the output video.")
    parser.add_argument("--quality", type=int, default=5, help="Quality of the output video (CRF value, lower is better).")

    args = parser.parse_args()
    main(args)




"""


folder = "A_V2V_ditto"

import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "1"
from huggingface_hub import HfApi
api = HfApi()

api.upload_folder(
    folder_path=os.path.join("/fsx-project/hanlin/outputs/V2V_baselines/ditto_832w_480h_121f/sstk_v2v_eval_304"),
    path_in_repo=folder, # Upload to a specific folder
    repo_id="hanlincs/Bifrost-2_debug",
    repo_type="model"
)









"""






"""

cd /home/hanlin/Bifrost-2/eval/V2V

conda deactivate
conda deactivate
conda deactivate
conda deactivate

source /home/hanlin/miniconda3/bin/activate

conda activate blip3o_new

cd /home/hanlin/Bifrost-2/eval/V2V


torchrun --nproc_per_node=8 Nov9_batch_infer_ditto.py --start_idx 0 --end_idx 80

torchrun --nproc_per_node=8 Nov9_batch_infer_ditto.py --start_idx 80 --end_idx 160

torchrun --nproc_per_node=8 Nov9_batch_infer_ditto.py --start_idx 160 --end_idx 240

torchrun --nproc_per_node=8 Nov9_batch_infer_ditto.py --start_idx 240 --end_idx 304


"""



"""

SSH_KEY=id_rsa_pem_format
chmod 600 /home/hanlin/.ssh/$SSH_KEY
eval "$(ssh-agent -s)"
ssh-add /home/hanlin/.ssh/$SSH_KEY
ssh -o StrictHostKeyChecking=no git@github.com


cd /home/hanlin
[ -d "./Ditto" ] && rm -rf "./Ditto"
git clone git@github.com:HL-hanlin/ditto.git
cd /home/hanlin/Ditto

"""
