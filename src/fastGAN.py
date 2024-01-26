import argparse
from pathlib import Path
from fastai.vision.gan import *
from fastai.callback.all import *

# Argument parsing
parser = argparse.ArgumentParser("Simple FastAI GAN example")
parser.add_argument("--path", type=Path, required=True, help="Path to training images")
parser.add_argument("--model_path", type=Path, default=Path('./models'), help="Path for model exports")
parser.add_argument("--save_every", type=int, default=10, help="Save model every N cycles")
parser.add_argument("--cycle_count", type=int, default=10, help="Number of training cycles")
parser.add_argument("--image_resolution", type=int, default=256, help="Training image resolution")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--autosave", action='store_true', help="Automatically save models")
args = parser.parse_args()

# Validation
if not args.path.exists():
    raise ValueError(f"Training path {args.path} does not exist.")
if not args.model_path.exists():
    args.model_path.mkdir(parents=True)

# Data loading
dblock = DataBlock(
    blocks=(TransformBlock, ImageBlock),
    get_x=generate_noise,
    get_items=get_image_files,
    splitter=RandomSplitter(),
    item_tfms=Resize(args.image_resolution, method=ResizeMethod.Crop),
    batch_tfms=Normalize.from_stats(torch.tensor([0.5, 0.5, 0.5]), torch.tensor([0.5, 0.5, 0.5]))
)
dls = dblock.dataloaders(args.path, bs=args.batch_size)

# GAN setup
generator = basic_generator(64, n_channels=3, n_extra_layers=1)
critic = basic_critic(64, n_channels=3, n_extra_layers=1, act_cls=partial(nn.LeakyReLU, negative_slope=0.2))
learn = GANLearner.wgan(dls, generator, critic)

# Callbacks
callbacks = [
    EarlyStoppingCallback(monitor='valid_loss', min_delta=0.1, patience=5),
    ReduceLROnPlateau(monitor='valid_loss', min_delta=0.1, patience=2)
]

# Training with autosave
print("Starting training...")
for epoch in range(args.cycle_count):
    learn.fit_one_cycle(1, wd=0., cbs=callbacks)
    if args.autosave and (epoch + 1) % args.save_every == 0:
        learn.save(args.model_path / f'model_epoch_{epoch+1}')
        print(f"Model saved at epoch {epoch+1}")

# Save final model
learn.save(args.model_path / 'final_model')
print("Training Complete.")
