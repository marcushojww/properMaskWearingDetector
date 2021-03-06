# specify path to the flowers and mnist dataset
MASK_DATASET_PATH = "dataset"
WITHOUT_MASK_PATH = "dataset/without_mask"
CORRECT_MASK_PATH = "dataset/with_mask"
INCORRECT_MASK_PATH = "dataset/incorrect_mask"

# specify the paths to our training and validation set 
TRAIN = "train"
VAL = "val"
TEST = "test"
PREDICT = "predict"

# set the input height and width
INPUT_HEIGHT = 128
INPUT_WIDTH = 128

# set the validation data split
VAL_SPLIT = 0.1
TEST_SPLIT = 0.2

# define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 20