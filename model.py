#/usr/bin/env python3
#
# PROGRAMMER: Robert C.
# DATE CREATED: 2019-02-23
# REVISED DATE:
# PURPOSE: A class which implements a pretrained models from torch vision 
#          and replaces its existing classfier with a custom network. 
#          It's main use is to predict flower classes.
#          - Train the network with custom hyperparameters
#          - Save and load checkpoints
#          - Predict class of an input image
##

import os
import numpy as np
from datetime import datetime
from PIL import Image

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


class Network(object):
    def __init__(self, output_size, arch="vgg16", hidden_layers=[1024]):
        """ Builds a classifier with arbitrary hidden layers which
            takes input from a pretrained model
        """
        super().__init__()

        # Load pretrained feature network
        self.model = models.__dict__[arch](pretrained=True)
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Get feature input size from original classifier
        old_classifier = list(self.model.children())[-1]
        
        if type(old_classifier) == nn.Linear:
            input_features = old_classifier.in_features
        else:
            for seq in range(len(old_classifier)):
                if type(old_classifier[seq]) == nn.Linear:
                    input_features = old_classifier[seq].in_features
                    break
     
        # Add the first layer
        classifier = nn.ModuleList([
            nn.Linear(input_features, hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(),
            ])
        
        # Add a variable number of more hidden layers + Relu, Dropout
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        for h1, h2 in layer_sizes:
            classifier.extend([
                nn.Linear(h1, h2),
                nn.ReLU(),
                nn.Dropout(),
                ])
        
        # Add output layer
        classifier.extend([nn.Linear(hidden_layers[-1], output_size), nn.LogSoftmax(dim=1)])
        
        # Replace classifier
        self.model.classifier = nn.Sequential(*classifier)

        # Append / Initialize parameters
        self.feature_arch = arch
        self.output_size = output_size
        self.criterion = nn.NLLLoss()
        self.hidden_layers = hidden_layers
        self.optimizer = None
        self.epochs = 0
    
    def save_checkpoint(self, file_path):
        """ Save model as a checkpoint along with associated parameters """
        
        checkpoint = {
              "feature_arch": self.feature_arch,
              "output_size": self.output_size,
              "hidden_layers": self.hidden_layers,
              "epochs": self.epochs,
              "optimizer": self.optimizer,
              "class_to_idx": self.model.class_to_idx,
              "state_dict": self.model.state_dict()
              }
        
        torch.save(checkpoint, file_path)
    
    def load_checkpoint(file_path):
        """ Rebuilds a model based on a checkpoint and returns it """
        
        checkpoint = torch.load(file_path)
        
        model = Network(checkpoint["output_size"], 
                        arch=checkpoint["feature_arch"], 
                        hidden_layers=checkpoint["hidden_layers"])
        model.model.load_state_dict(checkpoint["state_dict"])
        model.epochs = checkpoint["epochs"]
        model.optimizer = checkpoint["optimizer"]
        model.class_to_idx = checkpoint["class_to_idx"]
        
        return model
        
    def predict(self, image_path, topk=5, gpu=False):
        ''' Predict the class (or classes) of an image using this model
        '''
        
        # Check for GPU
        device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Load and process image
        image = process_image(Image.open(image_path))
        image.unsqueeze_(0)
        image = image.to(device)
        
        # Predict class
        self.model.eval()
        with torch.no_grad():
            output = self.model(image)

        # Get topk probabilities and classes
        prediction = F.softmax(output.data, dim=1).topk(topk)
        probs = prediction[0].data.cpu().numpy().squeeze()
        classes = prediction[1].data.cpu().numpy().squeeze()

        # Get actual class labels
        inverted_dict = dict([[self.class_to_idx[k], k] for k in self.class_to_idx])
        if topk > 1:
            classes = [inverted_dict[k] for k in classes]
        else:
            classes = inverted_dict[classes.item()]
        
        return probs, classes
    
    def validation(self, testloader, gpu=False):
        """ Validate network with test data and return loss and accuracy """
        
        # Check for GPU
        device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        loss = 0
        accuracy = 0
        
        self.model.eval()
        for images, labels in testloader:
            
            images = images.to(device)
            labels = labels.to(device)

            output = self.model.forward(images)
            loss += self.criterion(output, labels).item()

            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
        
        return loss, accuracy
    
    def train(self, data_dir, epochs=5, learning_rate=0.001, gpu=False, print_every=20):
        """ Trains the network 
            During training it outputs the validation loss and accuracy after
            After training a validation with the testset done
        """

        # Define image paths
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'
        
        # Define your transforms for the training, validation, and testing sets
        data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(
                                                    (0.485, 0.456, 0.406),
                                                    (0.229, 0.224, 0.225))
                                              ])

        # Load the datasets with ImageFolder and save class_to_idx
        image_datasets = {
            "train": datasets.ImageFolder(train_dir, transform=data_transforms),
            "valid": datasets.ImageFolder(valid_dir, transform=data_transforms),
            "test": datasets.ImageFolder(test_dir, transform=data_transforms)
            }
            
        self.model.class_to_idx = image_datasets['train'].class_to_idx
        
        # Using the image datasets and the trainforms, define the dataloaders
        image_loader = {
            "train": torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
            "valid": torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True),
            "test": torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True)
            }
        
        # Set Hyperparameters
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=learning_rate)
        total_epochs = self.epochs + epochs
        
        # Set hardware for training
        device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
        self.model.to(device)

        print("Start Training...")
        print("Device:", device)
        
        # Go through epochs
        steps = 0
        
        for e in range(1, epochs+1):
            self.model.train()
            running_loss = 0
            
            # Train network
            for images, labels in image_loader['train']:
                steps += 1
            
                images = images.to(device)
                labels = labels.to(device)
                
                self.optimizer.zero_grad()
                
                # Forward and backward passes
                outputs = self.model.forward(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                
                # Print validation loss and accuracy
                if steps % print_every == 0:
                    # Validation pass
                    self.model.eval()
                    with torch.no_grad():
                        vloss, vaccuracy = self.validation(image_loader['valid'], gpu=gpu)
                    
                    # Print progress
                    print("[{}]".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                          "Epoch: {}/{} |".format(self.epochs+e, total_epochs),
                          "Training Loss: {:.3f} |".format(running_loss/print_every),
                          "Validation Loss: {:.3f} |".format(vloss/len(image_loader['valid'])),
                          "Validation Accuracy: {:.1f}%".format(100*vaccuracy/len(image_loader['valid'])))
                        
                    running_loss = 0
                    self.model.train()
        
        # Set total epochs
        self.epochs = total_epochs
        
        # Validate network on test set
        self.model.eval()
        with torch.no_grad():
            _, accuracy = self.validation(image_loader['test'], gpu=gpu)
            
        print("Accuracy of the network on the test images: {:.2f}%"
              .format(100*accuracy / len(image_loader['test'])))
        

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a tensor
    '''
    
    # Resize image
    if image.size[0] < image.size[1]:
        image.thumbnail((256, image.size[1]))
    else:
        image.thumbnail((image.size[0], 256))
    
    # Crop image
    image = image.crop((
                image.size[0] / 2 - 112,
                image.size[1] / 2 - 112,
                image.size[0] / 2 + 112,
                image.size[1] / 2 + 112
                ))
    
    # Convert image to numpy array and convert color channel values
    np_image = np.array(image) / 256.
    
    # Normalize image array
    np_image[:, :, 0] = (np_image[:, :, 0] - 0.485) / 0.229
    np_image[:, :, 1] = (np_image[:, :, 1] - 0.456) / 0.224
    np_image[:, :, 2] = (np_image[:, :, 2] - 0.406) / 0.225
    
    # Transpose image array
    np_image = np_image.transpose(2, 0, 1)
    
    # Returns image array as a tensor
    return torch.from_numpy(np_image).float()
