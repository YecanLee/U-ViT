conda create -n uvit python==3.10
conda activate uvit

pip install gdown
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116  # install torch-1.13.1
pip install accelerate==0.12.0 absl-py ml_collections einops wandb ftfy==6.1.1 transformers==4.23.1
pip install timm==0.4.12
pip install lightning 

# xformers is optional, but it would greatly speed up the attention computation.
pip install -U xformers
pip install -U --pre triton
gdown 13StUdrjaaSXjfqqF7M47BzPyhMAArQ4u

# uncomment the following line if you want to use the pretrained model with 512 x 512 resolution
# gdown 1uegr2o7cuKXtf2akWGAN2Vnlrtw5YKQq 

gdown 10nbEiFd4YCHlzfTkJjZf45YcSMCN34m6