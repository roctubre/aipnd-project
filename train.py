#/usr/bin/env python3
#
# PROGRAMMER: Robert C.
# DATE CREATED: 2019-02-23
# REVISED DATE:
# PURPOSE: Implements model.py for training
##

import torch
from torchvision import models
import argparse
from model import Network
import os
import sys


supported_archs = ["alexnet", "vgg16", "densenet161"]


def parse_arguments():
    """ Checks and returns arguments passed to the program """
    
    # Create parser and arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', action="store")
    parser.add_argument('--save_dir', action='store',
                        default="",
                        help='Set directory to save checkpoints')
    parser.add_argument('--arch', action='store',
                        default="densenet161",
                        help='Architecture for input features: '
                        + str([arch for arch in supported_archs]))
    parser.add_argument('--learning_rate', action='store',
                        default=0.001, type=float,
                        help='Learning rate for training')
    parser.add_argument('--epochs', action='store',
                        default=5, type=int,
                        help='Training iteration count')
    parser.add_argument('--hidden_units', action='append',
                        type=int,
                        dest='hidden_layers',
                        help='Add a hidden layer with X units')
    parser.add_argument('--gpu', action='store_true',
                        default=False,
                        help='Use GPU for calculations')
    
    # Try to parse arguments
    try:
        results = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))
    
    # Check data directory
    if not os.path.isdir(results.data_directory):
        parser.error("\"{}\" is not a valid directory".format(results.data_directory))
    
    # Check save directory
    if not results.save_dir == "" and not os.path.isdir(results.save_dir):
        parser.error("\"{}\" is not a valid directory".format(results.save_dir))   
    
    # Check if valid architecture
    if results.arch not in supported_archs:
        parser.error("\"{}\" is not a valid architecture.".format(results.arch))
    
    # Check if GPU is available
    if results.gpu and not torch.cuda.is_available():
        parser.error("GPU can't be used in this environment")
        
    # Check hidden layers
    if not results.hidden_layers:
        results.hidden_layers = [1024]
    
    return results


if __name__ == "__main__":
    # Get arguments
    args = parse_arguments()
    
    # Get number of classes
    class_count = len(next(os.walk(args.data_directory + "/train"))[1])
    
    # Output parameters
    print("##### Training parameters #####",
          "Feature model:\t{}".format(args.arch),
          "Hidden layers:\t{}".format(args.hidden_layers),
          "Learning rate:\t{}".format(args.learning_rate),
          "Train Epochs:\t{}".format(args.epochs),
          "GPU Usage:\t{}".format(args.gpu),
          sep="\n")
    
    # Create model
    model = Network(class_count, args.arch, args.hidden_layers)
    model.train(args.data_directory,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                gpu=args.gpu)
    
    # Save model checkpoint to the defined save_dir
    # Use a filename not yet in use
    saveto = os.path.join(args.save_dir, "checkpoint_{}_0.pth".format(args.arch))
    counter = 1
    while os.path.isfile(saveto):
        saveto = os.path.join(args.save_dir, "checkpoint_{}_{}.pth".format(args.arch, counter))
    
    model.save_checkpoint(saveto)
    print("Checkpoint saved to: {}".format(saveto))
