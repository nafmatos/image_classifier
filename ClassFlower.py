# General libraries:
import os
import json
import numpy as np
import time
from collections import OrderedDict
from PIL import Image
import sys

# Pytorch library:
import torch
from torchvision import transforms, datasets, models
from torch import nn, optim

# Constants:
# Normalization paremeters used by the pre-trained network:
MEANS = [0.485, 0.456, 0.406]
STDEV = [0.229, 0.224, 0.225]

# Classifier class:

class ClassFlower():
    """
    Class to train and predict flower species.
    It can easily be tunned to predict other type of images.
    """

    def __init__(self):
        """
        Class initializer
        """
        self.NBATCH = 32

    def set_dir(self, data_dir):
        """
        Function to set the dataset directories, which are stored inside the data_dir folder.
        """
        self.data_map = dict()
        self.data_map['train'] = os.path.join(data_dir, 'train')
        self.data_map['test'] = os.path.join(data_dir, 'test')
        self.data_map['valid'] = os.path.join(data_dir, 'valid')

        # Check if the folders exists:
        for d in self.data_map.values():
            if not os.path.isdir(d):
                print("Folder '{}' not found.".format(d))
                sys.exit()

    def set_transforms(self):
        """
        Set the transforms for all datasets.
        """
        #
        self.data_transforms = dict()
        self.data_transforms['train'] = transforms.Compose([transforms.RandomRotation(30),
                                                            transforms.RandomResizedCrop(224),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(MEANS, STDEV)])
        #
        self.data_transforms['test'] = transforms.Compose([transforms.Resize(256),
                                                           transforms.CenterCrop(224),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize(MEANS, STDEV)])
        #
        self.data_transforms['valid'] = self.data_transforms['test']

    def load_datasets(self):
        """
        Load the datasets with ImageFolder.
        """
        self.image_datasets = dict()
        #
        for key, value in self.data_map.items():
            self.image_datasets[key] = datasets.ImageFolder(value,
                                                            transform=self.data_transforms[key])

    def set_dataloaders(self):
        """
        Define the dataloaders using image datasets and transforms.
        """
        self.dataloaders = dict()
        #
        for key in self.data_map:
            self.dataloaders[key] = torch.utils.data.DataLoader(self.image_datasets[key],
                                                                batch_size=self.NBATCH,
                                                                shuffle=True)
    def load_cat_map(self, cat_name_file):
        # Parse inputs from the mapping file:
        try:
            with open(cat_name_file, 'r') as f:
                self.cat_to_name = json.load(f)
        except:
            print("Could not load category to name map file: '{}".format(cat_name_file))
            sys.exit(cat_name_file)

    def choose_device(self, use_gpu):
        """
        Choose the device to run, based on the user choice and availability.
        """
        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("Proceeding with GPU.")
            else:
                print("GPU is not available, proceeding with CPU.")
                self.device = torch.device("cpu")
        else:
            print("Proceeding with CPU.")
            self.device = torch.device("cpu")

    def build(self):
        """
        Build the neural network.
        """
        # Choose the pretrained network from torch.vision.modules:
        if self.choice == "vgg16":
            self.model = models.vgg16(pretrained=True)
            ninp = 25088
        elif self.choice == "densenet121":
            self.model = models.densenet121(pretrained=True)
            ninp = 1024
        else:
            # There is a lot of room to future improvement here.
            print("Model {} not available.".format(self.choice))
            print("Only 'vgg16' and 'densenet121' are available.")
            print("Moving forward with 'vgg16'.")
            self.model = models.vgg16(pretrained=True)
            ninp = 25088
        #
        # Freeze the parameters of the pre-trained network, otherwise the algorithm will
        # backpropagate to them:
        #
        for parameters in self.model.parameters():
            parameters.requires_grad = False
        #
        # Set the classifier architecture according to the chosen parameters:
        #
        self.hidden_units = [ninp] + self.hidden_user + [self.ncat]
        # Hidden layers:
        clf_dict = OrderedDict()
        for i in range(len(self.hidden_units) - 2):
            clf_dict['h' + str(i)] = nn.Linear(self.hidden_units[i], self.hidden_units[i+1])
            clf_dict['f' + str(i)] = nn.ReLU()
            clf_dict['d' + str(i)] = nn.Dropout(self.dropout)
        # Output layer:
        clf_dict['hout'] = nn.Linear(self.hidden_units[i+1], self.ncat)
        clf_dict['fout'] = nn.LogSoftmax(dim=1)
        # Set the model classifier
        classifier = nn.Sequential(clf_dict)
        self.model.classifier = classifier

    def set_train_parameters(self, data_dir, save_dir, arch, hidden_user,
                             learning_rate, epochs, cat_name_file, dropout):
        """
        Inputs from train module:
        data_dir: Directory containing datasets.
        save_dir: Directory to save the checkpoint file.
        arch: Pretrained neural network: only 'vgg16' and 'densenet' are available.
        hidden_user: List of hidden units of the classifer layer.
        learning_rate: Learning Rate.
        epochs: Number of epochs.
        cat_name_file: JSON file containing the dictionary map from index to class name.
        dropout: Dropout used in the training phase.
        """
        self.set_dir(data_dir)
        self.checkpoint_file = os.path.join(save_dir, 'checkpoint.pth')
        self.choice = arch
        self.hidden_user = hidden_user
        self.learning_rate = learning_rate
        self.epochs = epochs        
        self.set_transforms()
        self.load_datasets()
        self.set_dataloaders()        
        self.load_cat_map(cat_name_file)
        self.ncat = len(self.cat_to_name)        
        self.dropout = dropout
        self.choose_device(True)        
        self.build() # Build the model

    def save_check_point(self):
        self.model.class_to_idx = self.image_datasets['train'].class_to_idx
        #
        model_dict = dict()
        model_dict['pre_trained'] = self.choice
        model_dict['hidden_user'] = self.hidden_user
        model_dict['ncat'] = self.ncat
        model_dict['model_state_dict'] = self.model.state_dict() # Includes the dropout
        #model_dict['optimizer_state_dict'] = optimizer.state_dict() # Includes the learning rate
        model_dict['class_to_idx'] = self.model.class_to_idx
        # Save the model to file.
        torch.save(model_dict, self.checkpoint_file)
        
    def train(self):
        """
        Train the model.
        """
        start = time.time() # Pick the initial time

        # Choose the loss function:
        criterion = nn.NLLLoss()

        # Set the optimizer. Take care to train only the classifier parameters.
        optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.learning_rate)

        # Copy the model to the available device (CPU or GPU):
        self.model.to(self.device)

        step = 0
        for epoch in range(self.epochs):
            running_loss = 0.0
            # Train the model classifier:
            for inputs, labels in self.dataloaders['train']:
                # Increment the current step:
                step += 1
                # Move inputs and labels to the available device (CPU or GPU):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Reset the optimizer to zero, otherwise it will accumulate:
                optimizer.zero_grad()
                # Feed forward the model
                log_prob = self.model.forward(inputs)
                # Evaluate the loss function:
                loss = criterion(log_prob, labels)
                # Backpropagate the loss:
                loss.backward()
                # Update the parameters:
                optimizer.step()
                # Update the loss:
                running_loss += loss.item()

                # Display validation loss and accuracy:
                if step % self.NBATCH == 0:
                    # Set the model to the evaluation mode:
                    self.model.eval()
                    # Initialize loss and accuracy:
                    valid_loss = 0.0
                    accuracy = 0.0

                    # Will not calculate gradients, so turn off the gradients to improve performance:
                    with torch.no_grad():
                        for inputs2, labels2 in self.dataloaders['valid']:
                            # Move inputs and labels to the available device (CPU or GPU):
                            inputs2, labels2 = inputs2.to(self.device), labels2.to(self.device)
                            # Feed forward the model:
                            log_prob2 = self.model.forward(inputs2)
                            # Evaluate the loss function:
                            batch_loss = criterion(log_prob2, labels2)
                            # Update the validation loss:
                            valid_loss += batch_loss.item()
                            # Since we are using the LogSoftmax function as output, we have to exponentiate
                            # the output to get the correct values:
                            probabilities = torch.exp(log_prob2)
                            # Get the top probability and classs
                            top_prob, top_class = probabilities.topk(1, dim=1)
                            # Get a tensor with 1 when class matches with prediction and 0 otherwise
                            equals = top_class == labels2.view(*top_class.shape)
                            # Update the accuracy of the validation data:
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    # Display the loss and accuracy on the validation set
                    # Rubric requires: "During training, the validation loss and accuracy are displayed."
                    print("Epoch: {:02d}".format(epoch+1), end='/')
                    print("{:02d}".format(self.epochs), end=', ')
                    print("Train loss: {:6.3f}".format(running_loss/self.NBATCH), end=', ')
                    print("Validation loss: {:6.3f}".format(valid_loss/len(self.dataloaders['valid'])), end=', ')
                    print("Validation accuracy: {:6.3f}".format(accuracy/len(self.dataloaders['valid'])), end=', ')

                    # Show elapsed time:
                    dt = time.time() - start
                    dt = dt/60
                    print("Elapsed Time: {:6.1f}min".format(dt))

                    # Set the model to training mode:
                    self.model.train()

                    # Zero the running_loss:
                    running_loss = 0.0
        
        # Save the check point file:
        self.save_check_point()

    def set_predict_parameters(self, file_name, check_point_file, topk, cat_name_file, use_gpu):
        """
        Parse the inputs from predict module:

        file_name: File name of the image to predict.
        check_point: Checkpoint file of the previous trained neural network.
        topk: Number of top predicted classes to show.
        cat_name_file: JSON file containing the dictionary map from index to class name.
        use_gpu: If included will run the prediction using GPU (if available), otherwise uses CPU.
        """
        # Parse inputs from user to the class:
        self.topk = topk
        self.image = file_name

        # Parse inputs from the check point file to the class:
        try:
            self.checkpoint = torch.load(check_point_file, map_location='cpu')
        except:
            print("Could not load checkpoint from file: '{}'.".format(check_point_file))
            print("Please check and try again.")
            sys.exit(check_point_file)
        else:
            self.choice = self.checkpoint["pre_trained"]
            self.hidden_user = self.checkpoint["hidden_user"]
            self.ncat = self.checkpoint["ncat"]
            # Set the dropout only to use in the build function,
            # but it will not make any effect in the prediction phase.
            self.dropout = 0.5
            self.build() # Build the model.
            self.model = self.model.to(torch.device('cpu'))
            self.model.load_state_dict(self.checkpoint["model_state_dict"])
            self.model.class_to_idx = self.checkpoint["class_to_idx"]

        # Parse inputs from the mapping file:
        self.load_cat_map(cat_name_file)
        print("If a predicted class is not available in '{}' it will be shown as 'NA'."\
            .format(cat_name_file))

        # Create a reversed dictionary from index to class:
        self.idx_to_class = dict(zip(
                                     self.model.class_to_idx.values(),
                                     self.model.class_to_idx.keys()
                                    )
                                )

        # Choose the device to be used (CPU or GPU):
        self.choose_device(use_gpu)

        # Set the data transforms:
        self.set_transforms()

    def process_image(self):
        """
        Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array.
        """
        try:
            pil_img = Image.open(self.image)
        except:
            print("Could not read the image file: '{}'".format(self.image))
            sys.exit()

        img_transform = self.data_transforms['test']
        self.torch_image = img_transform(pil_img)

    def predict(self):
        """
        Predict the class (or classes) of an image using a trained deep learning model.
        The image to predict should be stored in the input file (file_name).
        """
        # Process the image:
        self.process_image()
        self.torch_image = self.torch_image.unsqueeze_(0)

        # Move the model to the device
        self.model = self.model.to(self.device)
        self.torch_image = self.torch_image.to(self.device)

        # Set the evaluaton mode on
        self.model.eval()
        # Predict
        with torch.no_grad():
            log_pred = self.model.forward(self.torch_image)

            # Get the probabilities:
            probabilities = torch.exp(log_pred)

            # Get the top probability and classs
            self.top_prob, self.top_class = probabilities.topk(self.topk, dim=1)

            # Convert top classes to list:
            classlist = self.top_class.cpu().data.numpy()[0].tolist()
            classlist = [self.cat_to_name.get(self.idx_to_class.get(v, "NA"), "NA") for v in classlist]

            # Convert top probabilities to list:
            problist = self.top_prob.cpu().data.numpy().squeeze().tolist()

        return classlist, problist