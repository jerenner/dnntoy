import logging

# ---------------------------------------------------------------------------------------------------------
# USER INPUTS
# ---------------------------------------------------------------------------------------------------------
# Directory and run names
datdir = "/home/jrenner/dnn/data"                          # the base data directory
dname = "vox_dnn3d_NEXT100_Paolina222_v10x10x10_r200x200x200" # the data name
rdir = "/home/jrenner/dnn/run"                             # the run directory
rname = "dnn3d_NEXT100_Paolina222_v10x10x10_r200x200x200"     # the run name

read_googlenet = False                        # set to true only if using the GoogLeNet
train_init = True                             # if true, train from net with standard pre-training; if false, read in a previously trained net

# Input variables
vox_ext = 200          # extent of voxelized region in 1 dimension (in mm)
vox_size = 10           # voxel size (in mm)
vox_norm = 1.0         # voxel normalization

ntrain_evts = 10000    # number of training evts per dataset
nval_evts = 2000       # number of validation events
num_epochs = 30        # total number of epochs to train
epoch_blk_size = 1     # number of epochs to run per block (before reading new dataset); set equal to num_epochs unless data to be read in multiple blocks
dtblk_size = 10000      # number of signal and background events per training block
batch_size = 200       # training batch size

opt_lr = 1.0e-4        # optimizer learning rate
opt_eps = 1.0e-6       # optimizer epsilon (for AdamOptimizer)
opt_mom = 0.9          # optimizer momentum

log_to_file = True         # set to True to output log information to a file rather than the console
logging_lvl = logging.INFO  # logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL

plt_show = False       # show plots on-screen for dnnplot

# END USER INPUTS
# ------------------------------------------------------------------------------------------
