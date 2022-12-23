######### global settings  #########
GPU = True                                  # running on GPU is highly suggested
TEST_MODE = False                           # turning on the testmode means the code will run on a small dataset.
CLEAN = True                               # set to "True" if you want to clean the temporary large files after generating result
MODEL = 'vitstr'                          # model arch: vitstr, parseq, srn, abinet, trba, matrn
DATASET = 'str_xai'                       # model trained on: places365 or imagenet
QUANTILE = 0.1                            # the threshold used for activation
SEG_THRESHOLD = 0.001                        # the threshold used for visualization
SCORE_THRESHOLD = 0.001                      # the threshold used for IoU score (in HTML file)
TOPN = 10                                   # to show top N image with highest activation for each unit
PARALLEL = 1                                # how many process is used for tallying (Experiments show that 1 is the fastest)
# concept categories that are chosen to detect: "object", "part", "scene", "material", "texture", "color"
CATAGORIES = ["object"]
# OUTPUT_FOLDER = "result/pytorch_"+MODEL+"_"+DATASET # result will be stored in this folder
# MY_RESULT_DIRNAME = "Chars2Dataset"
DATASET_MAIN_DIR = "results_2char" ## See directory names in datasets/synthtigerseg (results_1char, results_2char, ...)
# OUTPUT_FOLDER = "results_dissect_netdissect/" + MODEL + "/" + DATASET_MAIN_DIR + "_"+MODEL+"_"+DATASET # result will be stored in this folder
LAYER_TO_EXTRACT = ["vitstr", "blocks", "11", "attn"]
OUTPUT_FOLDER = "results_dissect_net2vec/" + MODEL + "/" + DATASET_MAIN_DIR + "/pytorch_"+DATASET # result will be stored in this folder
SEGM__DIRS = "segm" # Used for checking the directory type if it is a segmentation image or rgb input image
### used for acquiring number of label.csv for a particular segmentation
SEGM_TO_LABEL_PKL = "/media/markytools/OrigDocs/markytools/Documents/MSEEThesis/STR/datasets/synthtigerseg/" + DATASET_MAIN_DIR + "/segToLabel.pickle"

### Do not edit finalDirName
finalDirName = ""
for layerModNames in LAYER_TO_EXTRACT:
    finalDirName += layerModNames+"."
finalDirName += "exp"
OUTPUT_FOLDER = OUTPUT_FOLDER + "/" + finalDirName ### Describe experiment

########### sub settings ###########
# In most of the case, you don't have to change them.
# DATA_DIRECTORY: where broaden dataset locates
# IMG_SIZE: image size, alexnet use 227x227
# NUM_CLASSES: how many labels in final prediction
# FEATURE_NAMES: the array of layer where features will be extracted
# MODEL_FILE: the model file to be probed, "None" means the pretrained model in torchvision
# MODEL_PARALLEL: some model is trained in multi-GPU, so there is another way to load them.
# WORKERS: how many workers are fetching images
# BATCH_SIZE: batch size used in feature extraction
# TALLY_BATCH_SIZE: batch size used in tallying
# INDEX_FILE: if you turn on the TEST_MODE, actually you should provide this file on your own

if DATASET == 'places365':
    NUM_CLASSES = 365
elif DATASET == 'imagenet':
    NUM_CLASSES = 1000
if MODEL == 'resnet18':
    FEATURE_NAMES = ['layer4']
    if DATASET == 'places365':
        MODEL_FILE = 'zoo/resnet18_places365.pth.tar'
        MODEL_PARALLEL = True
    elif DATASET == 'imagenet':
        MODEL_FILE = None
        MODEL_PARALLEL = False
elif MODEL == 'densenet161':
    FEATURE_NAMES = ['features']
    if DATASET == 'places365':
        MODEL_FILE = 'zoo/whole_densenet161_places365_python36.pth.tar'
        MODEL_PARALLEL = False
elif MODEL == 'resnet50':
    FEATURE_NAMES = ['layer4']
    if DATASET == 'places365':
        MODEL_FILE = 'zoo/whole_resnet50_places365_python36.pth.tar'
        MODEL_PARALLEL = False
elif MODEL == 'vitstr':
    FEATURE_NAMES = ['vitstr.blocks.4'] # Not really important except the len

if TEST_MODE:
    WORKERS = 1
    BATCH_SIZE = 4
    TALLY_BATCH_SIZE = 2
    TALLY_AHEAD = 1
    INDEX_FILE = 'index_sm.csv'
    OUTPUT_FOLDER += "_test"
else:
    WORKERS = 1
    BATCH_SIZE = 16
    TALLY_BATCH_SIZE = 16
    TALLY_AHEAD = 4
    INDEX_FILE = 'index.csv'
