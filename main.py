import argparse

#Create the parser
parser = argparse.ArgumentParser(description="Simple FastAI image classifier.")

#Add the arguments
parser.add_argument("--path", type=str, help="Path to the folder with subfolders named the image types, each containing training images")
parser.add_argument("--save_every", type=int, help="How often to save a model in cycles")
parser.add_argument("--cycle_count", type=int, help="Number of training cycles to use")
#parser.add_argument("--load_from", type=int, help="Save file to load from")
#parser.add_argument("--classify", action='store_const', const=True, default=False, help="Classify image from --image_path")
#parser.add_argument("--image_path", type=str, help="Path to an image to classify")
#parser.add_argument("--model", type=str, help="Resnet/Densnet model to use when training")

#Parse the arguments
args = parser.parse_args()

#Check input
if args.path == None:
    args.path = './'

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




from fastbook import *
from fastai.vision.all import *
setup_book()

path = Path(args.path)

fns = get_image_files(path)

#failed = verify_images(fns)

#failed.map(Path.unlink)

class DataLoaders(GetAttr):
    def __init__(self, *loaders): self.loaders = loaders
    def __getitem__(self, i): return self.loaders[i]
    train,valid = add_props(lambda i,self: self[i])

model = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))
dls = model.dataloaders(path)

model = model.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
dls = model.dataloaders(path)

learn = vision_learner(dls, resnet34, metrics=error_rate)

cycle = args.cycle_count
save_every = args.save_every

for x in range(cycle):
    learn.fine_tune(1)
    if x%save_every == 0:
        interp = ClassificationInterpretation.from_learner(learn)
        interp.plot_confusion_matrix()
        learn.export(str(path)+f'/export{x}.pkl')

if cycle%save_every != 0:
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    learn.export(str(path)+f'/export{cycle}.pkl')

print("\n\n\nTraining Complete.")
