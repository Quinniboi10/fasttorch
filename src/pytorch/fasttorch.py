import argparse
import logging
import os
from fastbook import *
from fastai.vision.all import *
from fastai.callback import *
from fastai.callback.tensorboard import TensorBoardCallback
from warnings import warn
import torchvision.models as models

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create the parser
parser = argparse.ArgumentParser(description="Simple FastAI image classifier.")

# Add the arguments
parser.add_argument("--path", type=str, default='./', help="Path to the folder with image data")
parser.add_argument("--model_path", type=str, default='./', help="Path for model exports")
parser.add_argument("--cycle_count", type=int, default=32, help="Number of training cycles")
parser.add_argument("--freeze_epochs", type=int, default=1, help="Number of freeze epochs")
parser.add_argument("--load_from", type=str, default=None, help="Cycle number or 'bestmodel' to load from")
parser.add_argument("--arch", type=str, help="Model architecture. Please refer to the fastai architecture lists")
parser.add_argument("--image_resolution", type=int, default=256, help="Image resolution for training")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--base_lr", type=float, default=2e-3, help="Base lr for the model to use")
parser.add_argument('--stop_early', action='store_true', default=False, help="Stop early when learning is not progressing")
parser.add_argument('--find_lr', action='store_true', default=False, help="Suggest learning rates with lr_find()")
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

# Handle model loading with error handling

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
    "xresnet50_deeper": xresnet50_deeper,
    "xresnet34_deeper": xresnet34_deeper,
    "xresnet18_deeper": xresnet18_deeper,
    "xresnet50_deep": xresnet50_deep,
    "xresnet34_deep": xresnet34_deep,
    "xresnet18_deep": xresnet18_deep,
    "xresnet152": xresnet152,
    "xresnet101": xresnet101,
    "xresnet50": xresnet50,
    "xresnet34": xresnet34,
    "xresnet18": xresnet18,
    "xse_resnext50_deeper": xse_resnext50_deeper,
    "xse_resnext34_deeper": xse_resnext34_deeper,
    "xse_resnext18_deeper": xse_resnext18_deeper,
    "xse_resnext50_deep": xse_resnext50_deep,
    "xse_resnext34_deep": xse_resnext34_deep,
    "xse_resnext18_deep": xse_resnext18_deep,
    "xsenet154": xsenet154,
    "xse_resnet152": xse_resnet152,
    "xresnext101": xresnext101,
    "xse_resnext101": xse_resnext101,
    "xse_resnet101": xse_resnet101,
    "xresnext50": xresnext50,
    "xse_resnext50": xse_resnext50,
    "xse_resnet50": xse_resnet50,
    "xresnext34": xresnext34,
    "xse_resnext34": xse_resnext34,
    "xse_resnet34": xse_resnet34,
    "xresnext18": xresnext18,
    "xse_resnext18": xse_resnext18,
    "xse_resnet18": xse_resnet18,
    "mobilenet_v2": mobilenet_v2,
    "mobilenet_v3_large": mobilenet_v3_large,
    "mobilenet_v3_small": mobilenet_v3_small,
    "alexnet": alexnet,
    "convnext_tiny": convnext_tiny,
    "convnext_small": convnext_small,
    "convnext_base": convnext_base,
    "convnext_large": convnext_large,
    "densenet121": densenet121,
    "densenet161": densenet161,
    "densenet169": densenet169,
    "densenet201": densenet201,
    "efficientnet_b0": efficientnet_b0,
    "efficientnet_b1": efficientnet_b1,
    "efficientnet_b2": efficientnet_b2,
    "efficientnet_b3": efficientnet_b3,
    "efficientnet_b4": efficientnet_b4,
    "efficientnet_b5": efficientnet_b5,
    "efficientnet_b6": efficientnet_b6,
    "efficientnet_b7": efficientnet_b7,
    "efficientnet_v2_s": efficientnet_v2_s,
    "efficientnet_v2_m": efficientnet_v2_m,
    "efficientnet_v2_l": efficientnet_v2_l,
    "googlenet": googlenet,
    "inception_v3": inception_v3,
    "regnet_y_400mf": regnet_y_400mf,
    "regnet_y_800mf": regnet_y_800mf,
    "regnet_y_1_6gf": regnet_y_1_6gf,
    "regnet_y_3_2gf": regnet_y_3_2gf,
    "regnet_y_8gf": regnet_y_8gf,
    "regnet_y_16gf": regnet_y_16gf,
    "regnet_y_32gf": regnet_y_32gf,
    "regnet_y_128gf": regnet_y_128gf,
    "regnet_x_400mf": regnet_x_400mf,
    "regnet_x_800mf": regnet_x_800mf,
    "regnet_x_1_6gf": regnet_x_1_6gf,
    "regnet_x_3_2gf": regnet_x_3_2gf,
    "regnet_x_8gf": regnet_x_8gf,
    "regnet_x_16gf": regnet_x_16gf,
    "regnet_x_32gf": regnet_x_32gf,
    "shufflenet_v2_x0_5": shufflenet_v2_x0_5,
    "shufflenet_v2_x1_0": shufflenet_v2_x1_0,
    "shufflenet_v2_x1_5": shufflenet_v2_x1_5,
    "shufflenet_v2_x2_0": shufflenet_v2_x2_0,
    "squeezenet1_0": squeezenet1_0,
    "squeezenet1_1": squeezenet1_1
}

path = Path(args.path)
fns = get_image_files(path)
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
        load_model(learn.model, os.path.join(args.model_path, "models", f"export {args.load_from} on {args.arch}.pth"))
        print("Exporting...")
        learn.path = Path(args.model_path)
        learn.export(os.path.join(args.model_path, "models", f"export {args.load_from} on {args.arch}.pkl"))
        print("Export complete.")
        exit(0)
    else:
        raise ValueError("Please pass a model to load from when exporting.")

if args.find_lr:
    print(f"\nRecommended base LR is {learn.lr_find()}")
    exit(0)

# Define callbacks
callbacks = [
    SaveModelCallback(fname=f"export bestmodel on {args.arch}"),
    ReduceLROnPlateau(min_delta=0.01, patience=2),
    TensorBoardCallback("./tensorboard/")
]
if args.stop_early: callbacks.append(EarlyStoppingCallback(min_delta=0.0025, patience=8))

# Load model if specified
if args.load_from is not None:
    load_model(learn.model, os.path.join(args.model_path, "models", f"export {args.load_from} on {args.arch}.pth"))

# Training
logging.info("Starting training...")
learn.fine_tune(args.cycle_count, base_lr=args.base_lr, freeze_epochs=args.freeze_epochs, cbs=callbacks)
logging.info("Training Complete.")
