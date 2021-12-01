# Responsible fpr dividing and structuring the dataset into a training and validation set
from imutils import paths
import numpy as np
import shutil
import os
import config

# load all the image paths and randomly shuffle them
print("[INFO] loading image paths...")
imagePaths = list(paths.list_images(config.MASK_DATASET_PATH))
withoutMaskImagePaths = list(paths.list_images(config.WITHOUT_MASK_PATH))
incorrectMaskImagePaths = list(paths.list_images(config.INCORRECT_MASK_PATH))
correctMaskImagePaths = list(paths.list_images(config.CORRECT_MASK_PATH))

print(len(withoutMaskImagePaths))
print(len(incorrectMaskImagePaths))
print(len(correctMaskImagePaths))
