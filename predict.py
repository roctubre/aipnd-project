#/usr/bin/env python3
#
# PROGRAMMER: Robert C.
# DATE CREATED: 2019-02-23
# REVISED DATE: 
# PURPOSE: Implements model.py for flower image prediction
##

import os
import argparse
import json

import torch
from torchvision import models

from model import Network


def parse_arguments():
    """Checks and returns arguments passed to the program"""
    parser = argparse.ArgumentParser()
    
    # Set arguments
    parser.add_argument("image_path", action="store")
    parser.add_argument("checkpoint", action="store")
    parser.add_argument("--category_names", action="store",
                        default="cat_to_name.json",
                        help="JSON file that maps class values to names")
    parser.add_argument("--top_k", action="store",
                        default=1, type=int,
                        help="Show top K predictions")
    parser.add_argument("--gpu", action="store_true",
                        default=False,
                        help="Use GPU for calculations")
        
    # Try to parse arguments
    try:
        results = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))
    
    # Check image path
    if not os.path.isfile(results.image_path):
        parser.error("\"{}\" is not a valid file".format(results.image_path))
    
    # Check checkpoint path
    if not os.path.isfile(results.checkpoint):
        parser.error("\"{}\" is not a valid file".format(results.checkpoint))
    
    # Check if GPU is available
    if results.gpu and not torch.cuda.is_available():
        parser.error("GPU can't be used in this environment")
    
    # Check top_k if none negative
    if results.top_k < 1:
        results.top_k = 1   

    return results
    

if __name__ == "__main__":
    # Get arguments
    args = parse_arguments()
    
    # Load class name mapping
    with open(args.category_names, "r") as f:
        cat_to_name = json.load(f)
    
    # Predict supplied image
    model = Network.load_checkpoint(args.checkpoint)
    probs, classes = model.predict(args.image_path, 
                                   gpu=args.gpu, 
                                   topk=args.top_k)

    # Print out top_k predictions
    if args.top_k == 1:
        print("Prediction: {}, Certainty: {:.1f}%".format(
                cat_to_name[classes], 
                100*probs))
    else:
        for i in range(len(classes)):
            print("#{} Prediction: {}, Certainty: {:.1f}%".format(
                    i+1,
                    cat_to_name[classes[i]],
                    100.*probs[i]))
