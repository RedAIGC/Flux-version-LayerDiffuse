import torch
import argparse
import os
import datetime
from lib_layerdiffuse.pipeline_flux_img2img import FluxImg2ImgPipeline
from lib_layerdiffuse.vae import TransparentVAE, pad_rgb
from PIL import Image
import numpy as np
from torchvision import transforms
from safetensors.torch import load_file
from PIL import Image, ImageDraw, ImageFont


def generate_img(pipe, trans_vae, args):
    original_image = (transforms.ToTensor()(Image.open(args.image))).unsqueeze(0)
    padding_feed = [x for x in original_image.movedim(1, -1).float().cpu().numpy()]
    list_of_np_rgb_padded = [pad_rgb(x) for x in padding_feed]
    rgb_padded_bchw_01 = torch.from_numpy(np.stack(list_of_np_rgb_padded, axis=0)).float().movedim(-1, 1).to(original_image.device)
    original_image_feed = original_image.clone()
    original_image_feed[:, :3, :, :] = original_image_feed[:, :3, :, :] * 2.0 - 1.0
    original_image_rgb = original_image_feed[:, :3, :, :] * original_image_feed[:, 3, :, :]

    original_image_feed = original_image_feed.to("cuda")
    original_image_rgb = original_image_rgb.to("cuda")
    rgb_padded_bchw_01 = rgb_padded_bchw_01.to("cuda")
    trans_vae.to(torch.device('cuda'))
    rng = torch.Generator("cuda").manual_seed(args.seed)

    initial_latent = trans_vae.encode(original_image_feed, original_image_rgb, rgb_padded_bchw_01, use_offset=True)

    latents = pipe(
        latents=initial_latent,
        image=original_image,
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        output_type="latent",
        generator=rng,
        guidance_scale=args.guidance,
        strength=args.strength,
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
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--prompt", type=str, default="a handsome man with curly hair, high quality")
    parser.add_argument(
        "--lora_weights",
        type=str,
        default="./models/layerlora.safetensors",
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--image", type=str, default="./imgs/causal_cut.png")
    args = parser.parse_args()
    
    pipe = FluxImg2ImgPipeline.from_pretrained(args.ckpt_path, torch_dtype=torch.bfloat16).to('cuda')
    pipe.load_lora_weights(args.lora_weights)

    trans_vae = TransparentVAE(pipe.vae, pipe.vae.dtype)
    trans_vae.load_state_dict(torch.load(args.trans_vae), strict=False)

    print("all loaded")

    img = generate_img(pipe, trans_vae, args)

    # save image
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    img.save(output_path)
