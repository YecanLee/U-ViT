import os
import argparse
import torch
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
import libs.autoencoder
from libs.uvit import UViT
import einops
from torchvision.utils import save_image
from PIL import Image
import lightning as l
from tqdm import tqdm, trange

# Define the noise scheduler
def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


# Define the model classifier free guidance and non-conditional model   
def model_fn(x, t_continuous, y, args): 
    _betas = stable_diffusion_beta_schedule()
    t = t_continuous * len(_betas)
    _cond = nnet(x, t, y=y)
    _uncond = nnet(x, t, y=torch.tensor([1000] * x.size(0), device=x.device))
    return _cond + args.cfg_scale * (_cond - _uncond)  
    
# classifier free guidance
# The original model_fn can only accept x and t_continuous as input. 
# We need to define a wrapper function to accept y as input. 
def create_model_fn(y, args):
    def model_fn_wrapper(x, t_continuous):
        return model_fn(x, t_continuous, y, args)
    return model_fn_wrapper


def load_models(args):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if args.image_size == 256:
        print(f"Using the pretrained-weights loading from {args.ckpt} for U-ViT-H/14 on ImageNet 256x256")
        z_size = 32
        patch_size = 2
    else:
        print(f"Using the pretrained-weights loading from {args.ckpt} for U-ViT-H/14 on ImageNet 512x512")
        z_size = 64
        patch_size = 4

    nnet = UViT(img_size=z_size,
                patch_size=patch_size,
                in_chans=4,
                embed_dim=1152,
                depth=28,
                num_heads=16,
                num_classes=1001,
                conv=False)

    nnet.to(device)
    nnet.load_state_dict(torch.load(f'imagenet{args.image_size}_uvit_huge.pth', map_location='cpu'))
    
    # Version check for torch.compile
    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    # Only use torch.compile if you are using PyTorch 2.0.x 
    if TORCH_MAJOR == 2:
        nnet = torch.compile(nnet)
    nnet.eval()

    autoencoder = libs.autoencoder.get_model('autoencoder_kl_ema.pth')
    autoencoder.to(device)
    if TORCH_MAJOR == 2:
        autoencoder = torch.compile(autoencoder)

    return nnet, autoencoder, z_size


def main(args):
    l.seed_everything(args.globa_seed)
    nnet, autoencoder, z_size = load_models(args)
    device = next(nnet.parameters()).device

    total_iterations = args.num_classes * args.num_samples_per_class
    batch_iterations = (total_iterations + args.batch_size - 1) // args.batch_size

    _betas = stable_diffusion_beta_schedule()  # set the noise schedule
    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

    all_samples = []

    for class_idx in trange(args.num_classes):
        class_samples = []
        remaining_samples = args.num_samples_per_class

        while remaining_samples > 0:
            current_batch_size = min(args.batch_size, remaining_samples)
            
            y = torch.full((current_batch_size,), class_idx, dtype=torch.long, device=device)

            z_init = torch.randn(current_batch_size, 4, z_size, z_size, device=device)
            model_fn_wrapper = create_model_fn(y, args)
            dpm_solver = DPM_Solver(model_fn_wrapper, noise_schedule, predict_x0=True, thresholding=False)

            with torch.inference_mode():
                with torch.cuda.amp.autocast():  # inference with mixed precision
                    z = dpm_solver.sample(z_init, steps=args.steps, eps=1. / len(_betas), T=1.)
                    samples = autoencoder.decode(z)

            samples = 0.5 * (samples + 1.)
            samples.clamp_(0., 1.)
            class_samples.append(samples)
            
            remaining_samples -= current_batch_size

        class_samples = torch.cat(class_samples, dim=0)
        all_samples.append(class_samples)

    all_samples = torch.cat(all_samples, dim=0)
    os.makedirs(args.save_path, exist_ok=True)
    
    total_images = args.num_classes * args.num_samples_per_class
    for i, sample in enumerate(all_samples):
        filename = f"{i:06d}.png"
        save_path = os.path.join(args.save_path, filename)
        save_image(sample.unsqueeze(0), save_path, nrow=1, padding=0)
    
    print(f"Saved {total_images} images to {args.save_path}")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image generation using DPM-Solver")
    parser.add_argument("--globa_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--samples_per_row", type=int, default=3, help="Number of samples per row")
    parser.add_argument("--cfg_scale", type=float, default=3.0, help="Classifier-free guidance scale")
    parser.add_argument("--steps", type=int, default=20, help="Number of sampling steps")
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256, help="Image size (256 or 512)")
    parser.add_argument("--ckpt", type=str, default="imagenet256_uvit_huge.pth", help="Path to the checkpoint file")
    parser.add_argument("--save_path", type=str, default="generated_samples", help="Path to save the generated images")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--num_samples_per_class", type=int, default=50, help="Number of samples per class")

    args = parser.parse_args()

    # Load models
    nnet, autoencoder, z_size = load_models(args)

    # Generate samples
    samples = main(args)

    print(f"Image generation complete. Results saved to {args.save_path}")
        