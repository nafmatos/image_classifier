"""
All functions were adapted from the following source:
https://stackoverflow.com/questions/14117415/in-python-using-argparse-allow-only-positive-integers
"""

import argparse
import os

def check_positive_int(value):
    """
    Function to check if the integers are positives.
    """
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("'{}' is an invalid positive integer value.".format(value))
    return ivalue

def check_positive_float(value):
    """
    Function to check if the floats are positives.
    """
    fvalue = float(value)
    if fvalue <= 0:
        raise argparse.ArgumentTypeError("'{}' is an invalid positive float value.".format(value))
    return fvalue

def check_dropout(value):
    """
    Function to check if the floats are between 0 and 1.
    """
    fvalue = float(value)
    if fvalue <= 0 or fvalue >= 1:
        raise argparse.ArgumentTypeError("'{}' should be >=0 and <=1.".format(value))
    return fvalue


def check_dir(value):
    """
    Function to check if the string passed is a valid folder.
    """
    if not os.path.isdir(value):
        raise argparse.ArgumentTypeError("'{}' is an invalid folder.".format(value))
    return value

def check_file(value):
    """
    Function to check if the string passed is a valid file.
    """
    if not os.path.exists(value):
        raise argparse.ArgumentTypeError("'{}' is an invalid file.".format(value))
    return value

def check_arch(value):
    """
    Function to check if the value passed belongs to a list.
    """
    if value not in ["vgg16", "densenet121"]:
        raise argparse.ArgumentTypeError("'{}' is invalid. It should be 'vgg16' or 'densenet121'.".format(value))
    return value