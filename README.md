# PnPInversion


This repository contains the implementation of the ICLR2024 paper "PnP Inversion: Boosting Diffusion-based Editing with 3 Lines of Code" 


## 🚀 Getting Started
<span id="getting-started"></span>

### Environment Requirement 🌍
<span id="environment-requirement"></span>

This is important!!! Since different models have different python environmnet requirements (e.g. diffusers' version), we list the environmnet in the folder "environment", detailed as follows:

- p2p_requirements.txt: for models in `run_editing_p2p.py`

    环境参考 `python==3.10.0` `environment/p2p_requirements.txt`
- masactrl_requirements.txt: for models in `run_editing_masactrl.py`
    环境参考 `python==3.10.0` `environment/masactrl_requirements.txt`
- pnp_requirements.txt: for models in `run_editing_pnp.py`
    环境参考 `python==3.9.20` `environment/pnp_requirements.txt`


#### 使用torchmetrics==1.6.3，无坑 
For example, if you want to use the models in `run_editing_p2p.py`, you need to install the environment as follows:

```shell
conda create -n p2p python=3.10.0 -y
conda activate p2p
pip install -r environment/p2p_requirements.txt
```
## 💖 Acknowledgement
<span id="acknowledgement"></span>

Our code is modified on the basis of [prompt-to-prompt](https://github.com/google/prompt-to-prompt), [StyleDiffusion](https://github.com/sen-mao/StyleDiffusion), [MasaCtrl](https://github.com/TencentARC/MasaCtrl), [pix2pix-zero](https://github.com/pix2pixzero/pix2pix-zero) , [Plug-and-Play](https://github.com/MichalGeyer/plug-and-play), [Edit Friendly DDPM Noise Space](https://github.com/inbarhub/DDPM_inversion), [Blended Latent Diffusion](https://github.com/omriav/blended-latent-diffusion), [Proximal Guidance](https://github.com/phymhan/prompt-to-prompt), [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix), thanks to all the contributors!

