import argparse
from fastbook import *
from fastai.vision.all import *
from fastai.callback import *
from warnings import warn

#Create the parser
parser = argparse.ArgumentParser(description="Simple FastAI image classifier.")

#Add the arguments
parser.add_argument("--path", type=str, default='./', help="Path to the folder with subfolders named the image types, each containing training images")
parser.add_argument("--model_path", type=str, default='./', help="Path to the folder for model exports")
parser.add_argument("--save_every", type=int, default=8, help="How often to save a model in cycles")
parser.add_argument("--cycle_count", type=int, default=32, help="Number of training cycles to use")
parser.add_argument("--load_from", type=str, default=None, help="Cycle number to load from")
parser.add_argument("--arch", type=str, choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet161', 'densenet169', 'densenet201'], help="Resnet/Densnet model to use when training")
parser.add_argument("--image_resolution", type=int, default=256, help="Resolution the training images will be rescaled to")
parser.add_argument("--batch_size", type=int, default=64, help="Default batch size")
parser.add_argument('--export', action='store_true',default=False, help="Create an .pkl export to be used for classification")
parser.add_argument('--autosave', action='store_true',default=False, help="Automatically save models")


#Parse the arguments
args = parser.parse_args()


try:
    args.load_from = int(args.load_from)
except:
    if args.load_from != 'bestmodel' and args.load_from != None:
        warn("Expected an integer or 'bestmodel'.")
        

if args.save_every == None:
    args.save_every = 8

if args.cycle_count == None:
    args.cycle_count = 64

#Prep the args
if args.path[len(args.path)-1] != '\\':
    if '\\' in args.path:
        args.path = args.path+'\\'

if args.path[len(args.path)-1] != '/':
    if '/' in args.path:
        args.path = args.path+'/'


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


# Save model after each epoch
def save_model(m, p): torch.save(m.state_dict(), p)

# Load model for resume training
def load_model(m, p): m.load_state_dict(torch.load(p))

def next_multiple(x, y):
    return (x + (y-1)) // y * y


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

strt = 0

learn = vision_learner(dls, arch_map[args.arch], metrics=error_rate, path=args.model_path)


if args.export:
    load_model(learn.model, str(args.model_path)+f'/models/export {args.load_from} on {args.arch}.pth')
    learn.export(str(args.model_path)+f'/models/export {args.load_from} on {args.arch}.pkl')
    exit()

if args.load_from != None:
    #learn.load(str(args.model_path)+f'/export {args.load_from} on {args.arch}.pkl')
    print("Loading and preparing model...")
    if args.load_from == 'bestmodel':
        load_model(learn.model, str(args.model_path)+'/models/bestmodel.pth')
    else:
        strt = args.load_from//args.save_every
        cc = args.load_from%args.save_every
        cnt = args.load_from
        load_model(learn.model, str(args.model_path)+f'/models/export {args.load_from} on {args.arch}.pth')

        if args.load_from >= args.cycle_count:
            warn("There is nothing to train. If this is unexpected, please check the inputs")
            exit()
        if cc != 0 and args.load_from != args.cycle_count:
            print(f"Starting training on cycles {args.load_from}-{next_multiple(args.load_from, args.save_every)}")
            learn.fine_tune(next_multiple(args.load_from, args.save_every)-args.load_from)
            cnt = next_multiple(args.load_from, args.save_every)
        


callbacks = [
    SaveModelCallback(fname="bestmodel"),
    EarlyStoppingCallback(min_delta=0.1, patience=5),
    ReduceLROnPlateau(min_delta=0.1, patience=2)
]
        
        
if args.autosave:
    #if args.load_from != None:
        #print("Loading and preparing model...")
        #load_model(learn.model, str(args.model_path)+f'/models/export {args.load_from} on {args.arch}.pth')
    print("Starting training...")
    if args.load_from == None:
        args.load_from = 0
    learn.fine_tune(args.cycle_count, cbs=callbacks)
    print("\n\n\nTraning Complete.")
    exit()


cycle = args.cycle_count
save_every = args.save_every

for x in range(strt, cycle//save_every):
    print(f"Starting training on cycles {x*save_every}-{(x+1)*save_every}")
    learn.fine_tune(save_every)
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    cnt = x*save_every
    save_model(learn.model, str(args.model_path)+f'/models/export {(x+1)*save_every} on {args.arch}.pth')

if cycle%save_every != 0 and cnt != cycle:
    learn.fine_tune(cycle%save_every)
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    save_model(learn.model, str(args.model_path)+f'/models/export {cycle} on {args.arch}.pth') 

print("\n\n\nTraining Complete.")
