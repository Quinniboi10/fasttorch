import argparse
from fastbook import *
from fastai.vision.all import *

#Create the parser
parser = argparse.ArgumentParser(description="Simple FastAI image classifier.")

#Add the arguments
parser.add_argument("--model_path", type=str, default='./', help="Path to the model export")
parser.add_argument("--image_path", type=str, required=True, help="Path to an image to classify")

#Parse the arguments
args = parser.parse_args()

#setup_book()
learn = load_learner(str(args.model_path))

print("Thinking...")

pred,pred_idx,probs = learn.predict(Path(args.image_path))
prob = f'{probs[pred_idx]:.04f}'
prob = round(float(prob*100), 2)
print(f"Classified as {pred} with {prob}% confidence.")
