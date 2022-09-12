import argparse
import os

root_folder = os.path.join('C:\\', 'Users', 'User', 'PycharmProjects', 'HandwrittenRecognition')

dataset = os.path.join(root_folder, 'dataset', 'a_z_handwritten_data.csv')

output_folder = os.path.join(root_folder, 'output')
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# initialize the number of epochs to train for, initial learning rate,
# batch size and image size
IMAGE_SIZE = 32
EPOCHS = 30
INIT_LR = 1e-1
BS = 128

early_stopping_patience = 3
cb_monitor = 'val_accuracy'
cb_mode = 'max'

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", help="path where to save each cnn output data")
args = vars(ap.parse_args())

cnn_output_folder = os.path.join(output_folder, args["output"])
if not os.path.exists(cnn_output_folder):
    os.mkdir(cnn_output_folder)
