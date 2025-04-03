import argparse
import json
import torch.nn.functional as F
import random
import os

from diffusers import DDIMScheduler

from models.p2p.inversion import DirectInversion
from models.masactrl.diffuser_utils import MasaCtrlPipeline
from models.masactrl.masactrl_utils import AttentionBase
from models.masactrl.masactrl_utils import regiter_attention_editor_diffusers
from models.masactrl.masactrl import MutualSelfAttentionControl

from torchvision.io import read_image
from datasets import get_dataloader
from torch.utils.data import DataLoader
from utils.utils import *
from utils.metrics import *

def mask_decode(encoded_mask,image_shape=[512,512]):
    length=image_shape[0]*image_shape[1]
    mask_array=np.zeros((length,))

    for i in range(0,len(encoded_mask),2):
        splice_len=min(encoded_mask[i+1],length-encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i]+j]=1

    mask_array=mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0,:]=1
    mask_array[-1,:]=1
    mask_array[:,0]=1
    mask_array[:,-1]=1

    return mask_array


def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_image(image_path, device,height,width):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (height,width))
    image = image.to(device)
    return image



class MasaCtrlEditor:
    def __init__(self, method_list, device, num_ddim_steps=30) -> None:
        self.device=device
        self.method_list=method_list
        self.num_ddim_steps=num_ddim_steps
        # init model
        self.scheduler = DDIMScheduler(beta_start=0.00085,
                                    beta_end=0.012,
                                    beta_schedule="scaled_linear",
                                    clip_sample=False,
                                    set_alpha_to_one=False)
        self.model = MasaCtrlPipeline.from_pretrained(args.model_path, scheduler=self.scheduler).to(device)
        self.model.scheduler.set_timesteps(self.num_ddim_steps)


    def __call__(self,
                edit_method,
                image_path,
                prompt_src,
                prompt_tar,
                guidance_scale,
                step=4,
                layper=10):
        if edit_method=="ddim+masactrl":
            return self.edit_image_ddim_MasaCtrl(image_path,prompt_src,prompt_tar,guidance_scale,step=step,layper=layper,height=args.height,width=args.width)
        elif edit_method=="directinversion+masactrl":
            return self.edit_image_directinversion_MasaCtrl(image_path,prompt_src,prompt_tar,guidance_scale,step=step,layper=layper)
        else:
            raise NotImplementedError(f"No edit method named {edit_method}")

    def edit_image_directinversion_MasaCtrl(self,image_path,prompt_src,prompt_tar,guidance_scale,step=4,layper=10):
        source_image=load_image(image_path, self.device)
        image_gt = load_512(image_path)

        prompts=["", prompt_tar]

        null_inversion = DirectInversion(model=self.model,
                                                num_ddim_steps=self.num_ddim_steps)

        _, image_enc_latent, x_stars, noise_loss_list = null_inversion.invert(
            image_gt=image_gt, prompt=prompts, guidance_scale=guidance_scale)
        x_t = x_stars[-1]

        # results of direct synthesis
        editor = AttentionBase()
        regiter_attention_editor_diffusers(self.model, editor)
        image_fixed = self.model([prompt_tar],
                            latents=x_t,
                            num_inference_steps=self.num_ddim_steps,
                            guidance_scale=guidance_scale,
                            noise_loss_list=None)

        # hijack the attention module
        editor = MutualSelfAttentionControl(step, layper)
        regiter_attention_editor_diffusers(self.model, editor)

        # inference the synthesized image
        image_masactrl = self.model(prompts,
                            latents= x_t.expand(len(prompts), -1, -1, -1),
                            guidance_scale=guidance_scale,
                            noise_loss_list=noise_loss_list)

        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

        # out_image=np.concatenate((
        #                         np.array(image_instruct),
        #                         ((source_image[0].permute(1,2,0).detach().cpu().numpy() * 0.5 + 0.5)*255).astype(np.uint8),
        #                         (image_masactrl[0].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8),
        #                         (image_masactrl[-1].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8)),1)
        #
        out_image = (image_masactrl[-1].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)

        return Image.fromarray(out_image)

    def edit_image_ddim_MasaCtrl(self, image_path,prompt_src,prompt_tar,guidance_scale,step=4,layper=10,height=512,width=512):
        source_image=load_image(image_path, self.device,height,width)

        prompts=["", prompt_tar]

        start_code, latents_list = self.model.invert(source_image,
                                            "",
                                            guidance_scale=guidance_scale,
                                            num_inference_steps=self.num_ddim_steps,
                                            return_intermediates=True,height=height,width=width)
        start_code = start_code.expand(len(prompts), -1, -1, -1)

        # results of direct synthesis
        editor = AttentionBase()
        regiter_attention_editor_diffusers(self.model, editor)
        image_fixed = self.model([prompt_tar],
                            latents=start_code[-1:],
                            num_inference_steps=self.num_ddim_steps,
                            guidance_scale=guidance_scale,height=height,width=width)

        # hijack the attention module
        editor = MutualSelfAttentionControl(step, layper)
        regiter_attention_editor_diffusers(self.model, editor)

        # inference the synthesized image
        image_masactrl = self.model(prompts,
                            latents=start_code,
                            guidance_scale=guidance_scale,
                            num_inference_steps=self.num_ddim_steps,height=height,width=width)

        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

        # out_image=np.concatenate((
        #                         np.array(image_instruct),
        #                         ((source_image[0].permute(1,2,0).detach().cpu().numpy() * 0.5 + 0.5)*255).astype(np.uint8),
        #                         (image_masactrl[0].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8),
        #                         (image_masactrl[-1].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8)),1)
        #
        out_image = (image_masactrl[-1].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        #out_image 512*512*3
        return Image.fromarray(out_image)


image_save_paths={
    "ddim+masactrl":"ddim+masactrl",
    # "directinversion+masactrl":"directinversion+masactrl",
    }


def main(args):


    import torchmetrics
    print(torchmetrics.__version__)
    if args.dtype == 'bfloat16':
        DTYPE = torch.bfloat16
    elif args.dtype == 'float16':
        DTYPE = torch.float16
    elif args.dtype == 'float32':
        DTYPE = torch.float32
    else:
        raise ValueError(f"不支持的数据类型: {args.dtype}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    rerun_exist_images = args.rerun_exist_images
    edit_method_list = args.edit_method_list

    masactrl_editor = MasaCtrlEditor(edit_method_list,torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),num_ddim_steps = args.num_steps)
    metrics = metircs()


    # ******** Input processing **********
    if args.eval_datasets == '':
        img = Image.open(args.image_path)
        train_transforms = transforms.Compose(
                    [
                    transforms.Resize((args.height, args.width), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                    ]
                )

        img = train_transforms(img).unsqueeze(0)
        dataloader = [img, args.image_path,args.source_prompt, args.target_prompt]
    else:
        default_transform = transforms.Compose(
            [
            transforms.Resize((args.height, args.width), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
            ]
        )

        dataset = get_dataloader(args.eval_datasets,default_transform)
        dataloader = DataLoader(
            dataset,
            batch_size=1,          # 每批64个样本
            shuffle=False,           # 训练时打乱数据
            num_workers=8,          # 使用4个子进程加载数据
            pin_memory=True         # 如果使用GPU，可以加速数据传输
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mean_clip_score = 0
    mean_clip_v_score = 0
    mean_mse_score = 0
    mean_psnr_score = 0
    mean_lpips_score = 0
    mean_ssim_score = 0
    mean_dino_score = 0
    count = 0
    for img_float32, image_path, source_prompt, target_prompt in dataloader:
        original_prompt = source_prompt[0]
        editing_prompt = target_prompt[0]
        image_path = image_path[0]

        for edit_method in edit_method_list:
            output_filename = f"num_steps{args.num_steps}_guidance{args.guidance_scale}.png"
            output_path = os.path.join(args.output_dir, output_filename)

            print(f"editing image [{image_path}] with [{edit_method}]")
            setup_seed()
            torch.cuda.empty_cache()
            edited_image = masactrl_editor(edit_method,
                                           image_path=image_path,
                                           prompt_src=original_prompt,
                                           prompt_tar=editing_prompt,
                                           guidance_scale=args.guidance_scale,
                                           step=4,
                                           layper=10
                                           )
        base, ext = os.path.splitext(output_path)
        counter = 1
        while os.path.exists(output_path):
            output_path = f"{base}_{counter}{ext}"
            counter += 1
        edited_image.save(output_path)
        print(f"finish")
        out_latent_float32=transforms.Compose(
                    [
                    transforms.Resize((args.height, args.width), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                    ]
                    )(edited_image).unsqueeze(0).to(device)

        # evaluation  img, out均为[-1,1]
        # clip score
        img_float32 = img_float32.to(device)
        clip_score = metrics.clip_scores(out_latent_float32,target_prompt)
        print(f"==> clip-T score: {clip_score:.4f}")
        mean_clip_score += clip_score
        # clip v score`````````
        clip_v_score = metrics.clip_scores(img_float32,out_latent_float32)
        print(f"==> clip-I score: {clip_v_score:.4f}")
        mean_clip_v_score += clip_v_score
        # mse score
        mse_score = metrics.mse_scores(img_float32, out_latent_float32)
        print(f"==> mse score: {mse_score:.4f}")
        mean_mse_score += mse_score
        #psnr score
        psnr_score = metrics.psnr_scores(img_float32, out_latent_float32)
        print(f"==> psnr score: {psnr_score:.4f}")
        mean_psnr_score += psnr_score
        #lpips score
        lpips_score = metrics.lpips_scores(img_float32, out_latent_float32)
        print(f"==> lpips score: {lpips_score:.4f}")
        mean_lpips_score += lpips_score
        #ssim score
        ssim_score = metrics.ssim_scores(img_float32, out_latent_float32)
        print(f"==> ssim score: {ssim_score:.4f}")
        mean_ssim_score += ssim_score
        #dino score
        dino_score = metrics.dino_scores(img_float32, out_latent_float32)
        print(f"==> dino score: {dino_score:.4f}")
        mean_dino_score += dino_score
        count+=1

    print('######### Evaluation Results ###########')
    mean_clip_score = mean_clip_score / count
    print(f"==> clip-T score: {mean_clip_score:.4f}")
    mean_clip_v_score = mean_clip_v_score / count
    print(f"==> clip-I score: {mean_clip_v_score:.4f}")
    mean_mse_score = mean_mse_score / count
    print(f"==> mse score: {mean_mse_score:.4f}")
    mean_psnr_score = mean_psnr_score / count
    print(f"==> psnr score: {mean_psnr_score:.4f}")
    mean_lpips_score = mean_lpips_score / count
    print(f"==> lpips score: {mean_lpips_score:.4f}")
    mean_ssim_score = mean_ssim_score / count
    print(f"==> ssim score: {mean_ssim_score:.4f}")
    mean_dino_score = mean_dino_score / count
    print(f"==> dino score: {mean_dino_score:.4f}")
    print('#######################################')

if __name__ == "__main__":
    import torch

    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_exist_images', action= "store_true") # rerun existing images
    parser.add_argument('--model_path', type=str, default='/data/lyw/stable-diffusion-v1-4', help='预训练模型的路径')
    parser.add_argument('--output_dir', type=str, default="masa_outputs") # the editing category that needed to run
    parser.add_argument('--eval-datasets', type=str, default='PIE-Bench', help='选择要编辑的数据集：EditEval_v1, PIE-Bench')
    parser.add_argument('--num-steps', type=int, default=30, help='时间步长的数量')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='interpolated_denoise 的引导比例')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16', 'float32'],
                        help='计算的数据类型')
    parser.add_argument('--edit_category_list', nargs = '+', type=str, default=["0","1","2","3","4","5","6","7","8","9"]) # the editing category that needed to run
    # parser.add_argument('--eddatasetsit_method_list', nargs = '+', type=str, default=["ddim+masactrl","directinversion+masactrl"]) # the editing methods that needed to run
    parser.add_argument('--edit_method_list', nargs='+', type=str,default=["ddim+masactrl"])  # the editing methods that needed to run
    parser.add_argument('--height', type=int, default=512,
                        help='输出图像的高度')
    parser.add_argument('--width', type=int, default=512,
                        help='输出图像的宽度')

    args = parser.parse_args()
    main(args)

