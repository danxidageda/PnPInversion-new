
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionPipeline
import numpy as np
from PIL import Image
import os
import json
import random
import argparse
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
import torchvision.transforms as T

from utils.utils import txt_draw,load_512,latent2image
from torchvision.io import read_image
from datasets import get_dataloader
from torch.utils.data import DataLoader
from utils.utils import *
from utils.metrics import *


NUM_DDIM_STEPS = 30

def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def get_timesteps(scheduler, num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    return timesteps, num_inference_steps - t_start


class Preprocess(nn.Module):
    def __init__(self, device,model_key):
        super().__init__()

        self.device = device
        self.use_depth = False

        print(f'[INFO] loading stable diffusion...')
        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", 
                                                 torch_dtype=torch.float16).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder", revision="fp16",
                                                          torch_dtype=torch.float16).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", revision="fp16",
                                                         torch_dtype=torch.float16).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        print(f'[INFO] loaded stable diffusion!')


    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, device="cuda"):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latents = 1 / 0.18215 * latents
            imgs = self.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def load_img(self, image_path,height,width):
        image_pil = T.Resize(height)(Image.open(image_path).convert("RGB"))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image = T.ToTensor()(image_pil).unsqueeze(0).to(device)
        return image

    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.mean * 0.18215
        return latents

    @torch.no_grad()
    def ddim_inversion(self, cond, latent):
        latent_list=[latent]
        timesteps = reversed(self.scheduler.timesteps)
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(timesteps):
                cond_batch = cond.repeat(latent.shape[0], 1, 1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.unet(latent, t, encoder_hidden_states=cond_batch).sample

                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps
                latent_list.append(latent)
        return latent_list

    @torch.no_grad()
    def ddim_sample(self, x, cond):
        timesteps = self.scheduler.timesteps
        latent_list=[]
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(timesteps):
                    cond_batch = cond.repeat(x.shape[0], 1, 1)
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    alpha_prod_t_prev = (
                        self.scheduler.alphas_cumprod[timesteps[i + 1]]
                        if i < len(timesteps) - 1
                        else self.scheduler.final_alpha_cumprod
                    )
                    mu = alpha_prod_t ** 0.5
                    sigma = (1 - alpha_prod_t) ** 0.5
                    mu_prev = alpha_prod_t_prev ** 0.5
                    sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                    eps = self.unet(x, t, encoder_hidden_states=cond_batch).sample

                    pred_x0 = (x - sigma * eps) / mu
                    x = mu_prev * pred_x0 + sigma_prev * eps
                    latent_list.append(x)
        return latent_list

    @torch.no_grad()
    def extract_latents(self, num_steps, data_path,
                        inversion_prompt='',height=512,width=512):
        self.scheduler.set_timesteps(num_steps)

        cond = self.get_text_embeds(inversion_prompt, "")[1].unsqueeze(0)
        image = self.load_img(data_path,height,width)
        latent = self.encode_imgs(image)

        inverted_x = self.ddim_inversion(cond, latent)
        latent_reconstruction = self.ddim_sample(inverted_x[-1], cond)
        rgb_reconstruction = self.decode_latents(latent_reconstruction[-1])
        latent_reconstruction.reverse()
        return inverted_x, rgb_reconstruction, latent_reconstruction


def register_time(model, t):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 't', t)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 't', t)


def register_attention_control_efficient(model, injection_schedule):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads

            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            if not is_cross and self.injection_schedule is not None and (
                    self.t in self.injection_schedule or self.t == 1000):
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)

                source_batch_size = int(q.shape[0] // 3)
                # inject unconditional
                q[source_batch_size:2 * source_batch_size] = q[:source_batch_size]
                k[source_batch_size:2 * source_batch_size] = k[:source_batch_size]
                # inject conditional
                q[2 * source_batch_size:] = q[:source_batch_size]
                k[2 * source_batch_size:] = k[:source_batch_size]

                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)
            else:
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)
                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)

            v = self.to_v(encoder_hidden_states)
            v = self.head_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)

            return to_out(out)

        return forward

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}  # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)


def register_conv_control_efficient(model, injection_schedule):
    def conv_forward(self):
        def forward(input_tensor, temb):
            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if temb is not None:
                temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)
            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                source_batch_size = int(hidden_states.shape[0] // 3)
                # inject unconditional
                hidden_states[source_batch_size:2 * source_batch_size] = hidden_states[:source_batch_size]
                # inject conditional
                hidden_states[2 * source_batch_size:] = hidden_states[:source_batch_size]

            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            return output_tensor

        return forward

    conv_module = model.unet.up_blocks[1].resnets[1]
    conv_module.forward = conv_forward(conv_module)
    setattr(conv_module, 'injection_schedule', injection_schedule)

class PNP(nn.Module):
    def __init__(self, model_key,device="cuda"):
        super().__init__()
        self.device = device

        # Create SD models
        print('Loading SD model')

        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=torch.float16).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.scheduler.set_timesteps(args.num_steps, device=self.device)
        self.n_timesteps=args.num_steps
        print('SD model loaded')

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
        return text_embeddings

    @torch.no_grad()
    def decode_latent(self, latent):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latent = 1 / 0.18215 * latent
            img = self.vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)
        return img

    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def get_data(self,image_path,height,width):
        # load image
        image = Image.open(image_path).convert('RGB') 
        image = image.resize((height,width), resample=Image.Resampling.LANCZOS)
        image = T.ToTensor()(image).to(self.device)
        return image

    @torch.no_grad()
    def denoise_step(self, x, t,guidance_scale,noisy_latent):
        # register the time step and features in pnp injection modules
        latent_model_input = torch.cat(([noisy_latent]+[x] * 2))

        register_time(self, t.item())

        # compute text embeddings
        text_embed_input = torch.cat([self.pnp_guidance_embeds, self.text_embeds], dim=0)

        # apply the denoising network
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']

        # perform guidance
        _,noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # compute the denoising step with the reference model
        denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']
        return denoised_latent

    def init_pnp(self, conv_injection_t, qk_injection_t):
        self.qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        self.conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        register_attention_control_efficient(self, self.qk_injection_timesteps)
        register_conv_control_efficient(self, self.conv_injection_timesteps)

    def run_pnp(self,image_path,noisy_latent,target_prompt,guidance_scale=7.5,pnp_f_t=0.8,pnp_attn_t=0.5,height=512,width=512):
        
        # load image
        self.image = self.get_data(image_path,height,width)
        self.eps = noisy_latent[-1]

        self.text_embeds = self.get_text_embeds(target_prompt, "ugly, blurry, black, low res, unrealistic")
        self.pnp_guidance_embeds = self.get_text_embeds("", "").chunk(2)[0]
        
        pnp_f_t = int(self.n_timesteps * pnp_f_t)
        pnp_attn_t = int(self.n_timesteps * pnp_attn_t)
        self.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)
        edited_img = self.sample_loop(self.eps,guidance_scale,noisy_latent)
        
        return edited_img

    def sample_loop(self, x,guidance_scale,noisy_latent):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            # for i, t in enumerate(self.scheduler.timesteps, desc="Sampling"):
            for i, t in enumerate(self.scheduler.timesteps):
                x = self.denoise_step(x, t,guidance_scale,noisy_latent[-1-i])

            decoded_latent = self.decode_latent(x)
                
        return decoded_latent

###



def edit_image_ddim_PnP(
    model,
    image_path,
    prompt_src,
    prompt_tar,
    pnp,
    guidance_scale=7.5,
    height=512,
    width=512,
    num_steps = 30
):
    torch.cuda.empty_cache()
    image_gt = load_512(image_path,height,width)
    _, rgb_reconstruction, latent_reconstruction = model.extract_latents(data_path=image_path,
                                         num_steps=num_steps,
                                         inversion_prompt=prompt_src,
                                            height=height,
                                            width=width)

    edited_image=pnp.run_pnp(image_path,latent_reconstruction,prompt_tar,guidance_scale,height,width)
    # return Image.fromarray(np.concatenate((
    #     image_gt,
    #     np.uint8(255*np.array(rgb_reconstruction[0].permute(1,2,0).cpu().detach())),
    #     np.uint8(255*np.array(edited_image[0].permute(1,2,0).cpu().detach())),
    #     ),1))
    return Image.fromarray(np.uint8(255*np.array(edited_image[0].permute(1,2,0).cpu().detach())))



def edit_image_directinversion_PnP(
    model,
    image_path,
    prompt_src,
    prompt_tar,
    pnp,
    guidance_scale=7.5,
    image_shape=[512,512],
):
    torch.cuda.empty_cache()
    image_gt = load_512(image_path)
    inverted_x, rgb_reconstruction, _ = model.extract_latents(data_path=image_path,
                                         num_steps=NUM_DDIM_STEPS,
                                         inversion_prompt=prompt_src)

    edited_image=pnp.run_pnp(image_path,inverted_x,prompt_tar,guidance_scale)
    
    image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

    return Image.fromarray(np.concatenate((
        image_instruct,
        image_gt,
        np.uint8(np.array(latent2image(model=pnp.vae, latents=inverted_x[1].to(pnp.vae.dtype))[0])),
        np.uint8(255*np.array(edited_image[0].permute(1,2,0).cpu().detach())),
        ),1))


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

image_save_paths={
    "ddim+pnp":"ddim+pnp",
    "directinversion+pnp":"directinversion+pnp",
    }

def main(args):

    model_key = args.model_path
    toy_scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
    toy_scheduler.set_timesteps(args.num_steps)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    timesteps_to_save, num_inference_steps = get_timesteps(toy_scheduler, num_inference_steps=args.num_steps,
                                                           strength=1.0,
                                                           device=device)
    model = Preprocess(device, model_key=model_key)
    pnp = PNP(model_key)
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

    edit_method_list = args.edit_method_list
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
            if edit_method == "ddim+pnp":
                edited_image = edit_image_ddim_PnP(
                    model=model,
                    image_path=image_path,
                    prompt_src=original_prompt,
                    prompt_tar=editing_prompt,
                    guidance_scale=7.5,
                    height=args.height,
                    width=args.width,
                    pnp = pnp
                )
            elif edit_method == "directinversion+pnp":
                edited_image = edit_image_directinversion_PnP(
                    model=model,
                    image_path=image_path,
                    prompt_src=original_prompt,
                    prompt_tar=editing_prompt,
                    guidance_scale=7.5,
                    pnp=pnp
                )
            else:
                raise NotImplementedError(f"No edit method named {edit_method}")
            base, ext = os.path.splitext(output_path)
            counter = 1
            while os.path.exists(output_path):
                output_path = f"{base}_{counter}{ext}"
                counter += 1
            edited_image.save(output_path)
            print(f"finish")
            out_latent_float32 = transforms.Compose(
                [
                    transforms.Resize((args.height, args.width), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                ]
            )(edited_image).unsqueeze(0).to(device)
            # evaluation  img, out均为[-1,1]
            # clip score
            img_float32 = img_float32.to(device)
            clip_score = metrics.clip_scores(out_latent_float32, target_prompt)
            print(f"==> clip-T score: {clip_score:.4f}")
            mean_clip_score += clip_score
            # clip v score`````````
            clip_v_score = metrics.clip_scores(img_float32, out_latent_float32)
            print(f"==> clip-I score: {clip_v_score:.4f}")
            mean_clip_v_score += clip_v_score
            # mse score
            mse_score = metrics.mse_scores(img_float32, out_latent_float32)
            print(f"==> mse score: {mse_score:.4f}")
            mean_mse_score += mse_score
            # psnr score
            psnr_score = metrics.psnr_scores(img_float32, out_latent_float32)
            print(f"==> psnr score: {psnr_score:.4f}")
            mean_psnr_score += psnr_score
            # lpips score
            lpips_score = metrics.lpips_scores(img_float32, out_latent_float32)
            print(f"==> lpips score: {lpips_score:.4f}")
            mean_lpips_score += lpips_score
            # ssim score
            ssim_score = metrics.ssim_scores(img_float32, out_latent_float32)
            print(f"==> ssim score: {ssim_score:.4f}")
            mean_ssim_score += ssim_score
            # dino score
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_exist_images', action= "store_true") # rerun existing images
    parser.add_argument('--model_path', type=str, default='/data/lyw/stable-diffusion-v1-4', help='预训练模型的路径')
    parser.add_argument('--output_dir', type=str, default="pnp_outputs") # the editing category that needed to run
    parser.add_argument('--eval-datasets', type=str, default='EditEval_v1', help='选择要编辑的数据集：EditEval_v1, PIE-Bench')
    parser.add_argument('--num-steps', type=int, default=30, help='时间步长的数量')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='interpolated_denoise 的引导比例')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16', 'float32'],
                        help='计算的数据类型')
    parser.add_argument('--edit_category_list', nargs = '+', type=str, default=["0","1","2","3","4","5","6","7","8","9"]) # the editing category that needed to run
    # parser.add_argument('--eddatasetsit_method_list', nargs = '+', type=str, default=["ddim+masactrl","directinversion+masactrl"]) # the editing methods that needed to run
    parser.add_argument('--edit_method_list', nargs='+', type=str,default=["ddim+pnp"])  # the editing methods that needed to run
    parser.add_argument('--height', type=int, default=512,
                        help='输出图像的高度')
    parser.add_argument('--width', type=int, default=512,
                        help='输出图像的宽度')
###

    args = parser.parse_args()
    main(args)
