import argparse
import logging
import os
import pandas as pd
from fastai.tabular.all import *
from fastai.callback.tensorboard import TensorBoardCallback
from warnings import warn
import ast  # For safely evaluating the string representation of a list

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create the parser
parser = argparse.ArgumentParser(description="Simple FastAI tabular model.")

# Define arguments
parser.add_argument("--path", type=str, required=True, help="Path to the CSV file with tabular data")
parser.add_argument("--model_path", type=str, default='./', help="Path for model exports")
parser.add_argument("--cycle_count", type=int, default=32, help="Number of training cycles")
parser.add_argument("--load_from", type=str, default=None, help="Cycle number or 'bestmodel' to load from")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument('--export', action='store_true', help="Export model as .pkl")
parser.add_argument("--dependent_var", type=str, required=True, help="Name of the dependent variable (target)")
parser.add_argument("--arch", type=str, default="[200, 100]", help="Architecture of the learner as a list of integers (e.g., '[10, 20]')")
parser.add_argument('--stopearly', action='store_true', default=False, help="Stop early when learning is not progressing")

# Parse the arguments
args = parser.parse_args()

# Convert the arch argument from a string to a list of integers
try:
    arch = ast.literal_eval(args.arch)
    if not isinstance(arch, list) or not all(isinstance(n, int) for n in arch):
        raise ValueError("Architecture should be a list of integers.")
except (ValueError, SyntaxError):
    logging.error("Invalid architecture format. Please provide a list of integers (e.g., '[10, 20]').")
    exit(1)

# Validate path
def validate_path(path):
    if not os.path.exists(path):
        logging.error(f"Path does not exist: {path}")
        exit(20)

validate_path(args.path)
validate_path(args.model_path)

# Load the data
df = pd.read_csv(args.path)

# Automatically determine categorical and continuous column names
dependent_var = args.dependent_var
independent_vars = df.columns.drop(dependent_var)
cat_names = [col for col in independent_vars if df[col].dtype == 'object']
cont_names = [col for col in independent_vars if df[col].dtype != 'object']

# Preprocessing
procs = [Categorify, FillMissing, Normalize]

# TabularDataLoaders
splits = RandomSplitter(valid_pct=0.2)(range_of(df))
dls = TabularDataLoaders.from_df(df, path=args.path, procs=procs, cat_names=cat_names, cont_names=cont_names, y_names=dependent_var, splits=splits, bs=args.batch_size)

# Define callbacks
callbacks = [
    SaveModelCallback(fname=f"export bestmodel on arch {str(arch)}"),
    ReduceLROnPlateau(min_delta=0.01, patience=2),
    # TensorBoardCallback("./tensorboard/")
]
if args.stopearly:
    callbacks.append(EarlyStoppingCallback(min_delta=0.0025, patience=8))

# Tabular learner
learn = tabular_learner(dls, layers=arch, metrics=accuracy, path=args.model_path, cbs=callbacks)

# Model exporting logic
if args.export:
    if args.load_from is not None:
        print("Exporting...")
        learn.path = Path(args.model_path)
        learn.export(f"export {args.load_from}.pkl")
        print("Export complete.")
        exit(0)
    else:
        raise ValueError("Please pass a model to load from when exporting.")

# Training
logging.info("Starting training...")
learn.fine_tune(args.cycle_count, cbs=callbacks)
logging.info("Training Complete.")
