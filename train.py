#!/usr/bin/env python3

import argparse
from check import check_positive_int, check_positive_float, check_dir, check_arch, check_file, check_dropout
from ClassFlower import ClassFlower

parser = argparse.ArgumentParser(description='Image Classifer - Training Module')

#

# Nonoptional arguments:
parser.add_argument("data_dir",
                    action="store",
                    help="Directory containing datasets.",
                    type=check_dir)

# Optional arguments:
parser.add_argument("--save_dir",
                    dest="save_dir",
                    action="store",
                    help="Directory to save the checkpoint file. Default='temp'.",
                    type=check_dir,
                    required=False,
                    default="temp")

parser.add_argument("--arch",
                    dest="arch",
                    action="store",
                    help="Pretrained neural network: only 'vgg16' and 'densenet' are available. Default='vgg16'.",
                    type=check_arch,
                    required=False,
                    default="vgg16")

parser.add_argument("--hidden_units",
                    dest="hidden_units",
                    action="store",
                    help="List of hidden units of the classifer layer. Default= 4096, 1024.",
                    type=check_positive_int,
                    required=False,
                    nargs='+',
                    default=[4096, 1024])

parser.add_argument("--learning_rate",
                    dest="learning_rate",
                    action="store",
                    help="Learning Rate. Default=0.001.",
                    type=check_positive_float,
                    required=False,
                    default=0.001)

parser.add_argument("--epochs",
                    dest="epochs",
                    action="store",
                    help="Number of epochs. Default=12",
                    type=check_positive_int,
                    required=False,
                    default=12)

parser.add_argument("--category_names",
                    dest="cat_name_file",
                    action="store",
                    help="JSON file containing the dictionary map from index to class name. Default=cat_to_name.json.",
                    type=check_file,
                    required=False,
                    default="cat_to_name.json")

parser.add_argument("--dropout",
                    dest="dropout",
                    action="store",
                    help="Dropout: >=0 and <=1. Default=0.5",
                    type=check_dropout,
                    required=False,
                    default=0.5)

def main():
    args = parser.parse_args()
    classifier = ClassFlower()
    classifier.set_train_parameters(args.data_dir,
                                    args.save_dir,
                                    args.arch,
                                    args.hidden_units,
                                    args.learning_rate,
                                    args.epochs,
                                    args.cat_name_file,
                                    args.dropout)
    classifier.train()
    print("See check point file 'checkpoint.pth' in '{}' folder.".format(args.save_dir))

if __name__ == "__main__":
    main()
