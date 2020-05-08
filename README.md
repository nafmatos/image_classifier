# Image Classifer with Deep Learning
This project is part of Udacity Machine Learning Nanodegree Program (ND229).

A Python image classifer application was built to recognize different species of flowers. 
It can be trained on a dataset, then predict new images using the trained model.
___
## Environment
This project was run using conda as environment manager. <br>
You can quickly reproduce the environment using the `environment.yml` file provided in the `env` folder.<br>
See instructions 
[here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).
___
## Notebook application
The file `Image Classifier Project.ipynb` will guide you through all steps performed to train, test and implement an 
image classifier using Pytorch library. 
___
## Command line application
The deep neural network model built and trained on Jupyter Notebook was converted into an application that others can 
use. The application is a pair of Python scripts that run from the command line.

* `train.py` should be used to train the model. Run `python train.py -h` to see instructions about how to use it.
* `predict.py` should be used to make predictions, after you have trained your model. Run `python predict.py -h` to see
instructions about how to use it.
* `test_unit.ipynb` contains some tests which were performed to test `train.py` and `predict.py` files.




