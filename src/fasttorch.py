import argparse
from pathlib import Path
from fastai.vision.gan import *
from fastai.vision.all import *
from fastai.callback.all import *

parser = argparse.ArgumentParser("Simple FastAI GAN example")

# Add the arguments
parser.add_argument("--path", type=str, required=True,
                    help="Path to the folder with subfolders named the image types, each containing training images")
parser.add_argument("--model_path", type=str, default='./', help="Path to the folder for model exports")
parser.add_argument("--save_every", type=int, default=10, help="How often to save a model in cycles")
parser.add_argument("--cycle_count", type=int, default=10, help="Number of training cycles to use")
parser.add_argument("--load_from", type=str, default=None, help="Cycle number to load from")
parser.add_argument("--image_resolution", type=int, default=256,
                    help="Resolution the training images will be rescaled to")
parser.add_argument("--batch_size", type=int, default=64, help="Default batch size")
parser.add_argument('--export', action='store_true', default=False,
                    help="Create an .pkl export to be used for classification")
parser.add_argument('--autosave', action='store_true', default=False, help="Automatically save models")

args = parser.parse_args()

# Prepare the arguments
args.arch = 'GAN'
try:
    args.load_from = int(args.load_from)
except:
    if args.load_from != 'bestmodel' and args.load_from != None:
        warn("Expected an integer or 'bestmodel'.")

# Prep the args
if args.path[len(args.path) - 1] != '\\':
    if '\\' in args.path:
        args.path = args.path + '\\'

if args.path[len(args.path) - 1] != '/':
    if '/' in args.path:
        args.path = args.path + '/'

if args.model_path[len(args.model_path) - 1] != '\\':
    if '\\' in args.model_path:
        args.model_path = args.model_path + '\\'
if args.model_path[len(args.model_path) - 1] != '/':
    if '/' in args.model_path:
        args.model_path = args.model_path + '/'


# Define some basic functions
# Save model after each epoch
def save_model(m, p): torch.save(m.state_dict(), p)


# Load model for resume training
def load_model(m, p): m.load_state_dict(torch.load(p))


def next_multiple(x, y):
    return (x + (y - 1)) // y * y


bs = args.batch_size
size = args.image_resolution
strt = 0

path = Path(args.path)


class DataLoaders(GetAttr):
    def __init__(self, *loaders): self.loaders = loaders

    def __getitem__(self, i): return self.loaders[i]

    train, valid = add_props(lambda i, self: self[i])


dblock = DataBlock(blocks = (TransformBlock, ImageBlock),
                   get_x = generate_noise,
                   get_items = get_image_files,
                   splitter = IndexSplitter([]),
                   item_tfms=Resize(size, method=ResizeMethod.Crop),
                   batch_tfms = Normalize.from_stats(torch.tensor([0.5,0.5,0.5]), torch.tensor([0.5,0.5,0.5])))

path = Path(args.path)
dls = dblock.dataloaders(path, path=path, bs=bs)

generator = basic_generator(64, n_channels=3, n_extra_layers=1)
critic = basic_critic(64, n_channels=3, n_extra_layers=1, act_cls=partial(nn.LeakyReLU, negative_slope=0.2))

callbacks = [
    SaveModelCallback(fname="bestmodel"),
    EarlyStoppingCallback(min_delta=0.1, patience=5),
    ReduceLROnPlateau(min_delta=0.1, patience=2)
]

learn = GANLearner.wgan(dls, generator, critic)

learn.fit(1, wd=0.)

if args.load_from != None:
    print("Loading and preparing model...")
    if args.load_from == 'bestmodel':
        load_model(learn.model, str(args.model_path) + 'models/bestmodel.pth')
    else:
        strt = args.load_from // args.save_every
        cc = args.load_from % args.save_every
        cnt = args.load_from
        load_model(learn.model, str(args.model_path) + f'models/export {args.load_from} on {args.arch}.pth')

        if args.load_from >= args.cycle_count:
            warn("There is nothing to train. If this is unexpected, please check the inputs")
            exit()
        if cc != 0 and args.load_from != args.cycle_count:
            print(f"Starting training on cycles {args.load_from}-{next_multiple(args.load_from, args.save_every)}")
            learn.fit(next_multiple(args.load_from, args.save_every) - args.load_from, cbs=callbacks)
            cnt = next_multiple(args.load_from, args.save_every)

if args.autosave:
    print("Starting training...")
    if args.load_from == None:
        args.load_from = 0
    learn.fit(args.cycle_count, cbs=callbacks)
    print("\n\n\nTraning Complete.")
    exit()

cycle = args.cycle_count
save_every = args.save_every

for x in range(strt, cycle // save_every):
    print(f"Starting training on cycles {x * save_every}-{(x + 1) * save_every}")
    learn.fit(save_every)
    cnt = x * save_every
    save_model(learn.model, str(args.model_path) + f'models/export {(x + 1) * save_every} on {args.arch}.pth')

if cycle % save_every != 0 and cnt != cycle:
    learn.fit(cycle % save_every)
    save_model(learn.model, str(args.model_path) + f'models/export {cycle} on {args.arch}.pth')

print("\n\n\nTraining Complete.")
