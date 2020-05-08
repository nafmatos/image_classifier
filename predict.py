#!/usr/bin/env python3

import argparse
from check import check_positive_int, check_dir, check_file
from ClassFlower import ClassFlower

parser = argparse.ArgumentParser(description='Image Classifer - Prediction Module')

# Nonoptional arguments:
parser.add_argument("file_name",
                    action="store",
                    help="File Name of the image to predict.",
                    type=check_file)

parser.add_argument("check_point_file",
                    action="store",
                    help="Checkpoint file of the previous trained neural network.",
                    type=check_file)

# Optional arguments:
parser.add_argument("--top_k",
                    dest="topk",
                    action="store",
                    help="Number of top predicted classes to show. Default=5.",
                    type=check_positive_int,
                    required=False,
                    default=5)

parser.add_argument("--category_names",
                    dest="cat_name_file",
                    action="store",
                    help="JSON file containing the dictionary map from index to class name. Default=cat_to_name.json.",
                    type=check_file,
                    required=False,
                    default="cat_to_name.json")

parser.add_argument("--gpu",
                    dest="use_gpu",
                    action="store_true",
                    help="If included will run the prediction using GPU (if available), otherwise uses CPU.",
                    required=False,
                    default=False)

def main():
    args = parser.parse_args()
    classifier = ClassFlower()
    classifier.set_predict_parameters(args.file_name,
                                      args.check_point_file,
                                      args.topk,
                                      args.cat_name_file,
                                      args.use_gpu)

    classlist, problist = classifier.predict()

    print("List of top {} classes and probailities:".format(args.topk))

    for i, (c, p) in enumerate(zip(classlist, problist)):        
        print("Rank:{:d} - Propability: {:5.4f} - Predicted Class: {}".format(i, p, c), end=" ")
        if i == 0:
            print("<<<=== PREDICTED CLASS")
        else:
            print()

if __name__ == "__main__":
    main()
