import argparse
import logging
import os
from fastbook import *
from fastai.vision.all import *
from fastai.callback import *
from fastai.callback.tensorboard import TensorBoardCallback
from warnings import warn

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create the parser
parser = argparse.ArgumentParser(description="Simple FastAI image classifier.")

# Add the arguments
parser.add_argument("--path", type=str, default='./', help="Path to the folder with image data")
parser.add_argument("--model_path", type=str, default='./', help="Path for model exports")
parser.add_argument("--cycle_count", type=int, default=32, help="Number of training cycles")
parser.add_argument("--load_from", type=str, default=None, help="Cycle number or 'bestmodel' to load from")
parser.add_argument("--arch", type=str, choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet161', 'densenet169', 'densenet201'], help="Model architecture")
parser.add_argument("--image_resolution", type=int, default=256, help="Image resolution for training")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument('--stopearly', action='store_true', default=False, help="Stop early when learning is not progressing")
parser.add_argument('--export', action='store_true', default=False, help="Export model as .pkl")

# Parse the arguments
args = parser.parse_args()

# Validate paths
def validate_path(path):
    if not os.path.exists(path):
        logging.error(f"Path does not exist: {path}")
        exit(20)

validate_path(args.path)
validate_path(args.model_path)

# Convert load_from to int or 'bestmodel'
if args.load_from:
    try:
        args.load_from = int(args.load_from)
    except ValueError:
        if args.load_from != 'bestmodel':
            warn("Expected an integer or 'bestmodel'.")

# Create model directory if not exists
if not os.path.exists(os.path.join(args.model_path, "models")):
    os.makedirs(os.path.join(args.model_path, "models"))

# Handle model loading and saving with error handling
def save_model(m, p):
    try:
        torch.save(m.state_dict(), p)
        logging.info(f"Model saved to {p}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")

def load_model(m, p):
    if not os.path.exists(p):
        logging.warning(f"Model file not found: {p}")
        return
    try:
        m.load_state_dict(torch.load(p))
        logging.info(f"Model loaded from {p}")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        
        
arch_map = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'densenet121': densenet121,
    'densenet161': densenet161,
    'densenet169': densenet169,
    'densenet201': densenet201
}

path = Path(args.path)
fns = get_image_files(path)
def save_model(m, p): torch.save(m.state_dict(), p)
def load_model(m, p): m.load_state_dict(torch.load(p))
def next_multiple(x, y):
    return (x + (y-1)) // y * y

# Define the dataloader and the learner
class DataLoaders(GetAttr):
    def __init__(self, *loaders): self.loaders = loaders
    def __getitem__(self, i): return self.loaders[i]
    train,valid = add_props(lambda i,self: self[i])
model = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(args.image_resolution))
dls = model.dataloaders(path)
model = model.new(
    item_tfms=RandomResizedCrop(args.image_resolution, min_scale=0.5),
    batch_tfms=aug_transforms())
dls = model.dataloaders(path, bs=args.batch_size)
learn = vision_learner(dls, arch_map[args.arch], metrics=error_rate, path=args.model_path)



# Model exporting logic
if args.export:
    if args.load_from is not None:
        load_model(learn.model, os.path.join(args.model_path, f"export {args.load_from} on {args.arch}.pth"))
        print("Exporting...")
        learn.path = Path(args.model_path)
        learn.export(f"export {args.load_from} on {args.arch}.pkl")
        print("Export complete.")
        exit(0)
    else:
        raise ValueError("Please pass a model to load from when exporting.")
    

# Define callbacks
callbacks = [
    SaveModelCallback(fname=f"export bestmodel on {args.arch}"),
    ReduceLROnPlateau(min_delta=0.01, patience=2),
    TensorBoardCallback("./tensorboard/")
]
if args.stopearly: callbacks.append(EarlyStoppingCallback(min_delta=0.0025, patience=8))

# Load model if specified
if args.load_from is not None:
    load_model(learn.model, os.path.join(args.model_path, f"export {args.load_from} on {args.arch}.pth"))

# Training
logging.info("Starting training...")
learn.fine_tune(args.cycle_count, cbs=callbacks)
logging.info("Training Complete.")
