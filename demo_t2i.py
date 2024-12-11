import torch
import argparse
import os
import datetime
from diffusers import FluxPipeline
from lib_layerdiffuse.vae import TransparentVAE
from PIL import Image
import numpy as np

def generate_img(pipe, trans_vae, args):

    latents = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        output_type="latent",
        generator=torch.Generator("cuda").manual_seed(args.seed),
        guidance_scale=args.guidance,

    ).images

    latents = pipe._unpack_latents(latents, args.height, args.width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor

    with torch.no_grad():
        original_x, x = trans_vae.decode(latents)

    x = x.clamp(0, 1)
    x = x.permute(0, 2, 3, 1)
    img = Image.fromarray((x*255).float().cpu().numpy().astype(np.uint8)[0])

    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--trans_vae", type=str, default="./models/TransparentVAE.pth")
    parser.add_argument("--output_dir", type=str, default="./flux-layer-outputs")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="base dtype")
    parser.add_argument("--seed", type=int, default=11111)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--prompt", type=str, default="glass bottle, high quality")
    parser.add_argument(
        "--lora_weights",
        type=str,
        default="./models/layerlora.safetensors",
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    args = parser.parse_args()
    
    pipe = FluxPipeline.from_pretrained(args.ckpt_path, torch_dtype=torch.bfloat16).to('cuda')
    pipe.load_lora_weights(args.lora_weights)

    trans_vae = TransparentVAE(pipe.vae, pipe.vae.dtype)
    trans_vae.load_state_dict(torch.load(args.trans_vae), strict=False)
    trans_vae.to('cuda')

    print("all loaded")

    img = generate_img(pipe, trans_vae, args)

    # save image
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    img.save(output_path)
