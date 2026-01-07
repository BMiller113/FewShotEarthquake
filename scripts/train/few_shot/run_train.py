import argparse

from train import main

parser = argparse.ArgumentParser(description="Train prototypical networks (few-shot / xBD adaptation)")


# Data args (original + xBD)
default_dataset = "omniglot"
parser.add_argument("--data.dataset", type=str, default=default_dataset, metavar="DS",
                    help=f"data set name (default: {default_dataset})")

default_split = "vinyals"
parser.add_argument("--data.split", type=str, default=default_split, metavar="SP",
                    help=f"split name (default: {default_split})")

parser.add_argument("--data.way", type=int, default=60, metavar="WAY",
                    help="number of classes per episode (default: 60)")
parser.add_argument("--data.shot", type=int, default=5, metavar="SHOT",
                    help="number of support examples per class (default: 5)")
parser.add_argument("--data.query", type=int, default=5, metavar="QUERY",
                    help="number of query examples per class (default: 5)")

parser.add_argument("--data.test_way", type=int, default=5, metavar="TESTWAY",
                    help="number of classes per episode in test. 0 means same as data.way (default: 5)")
parser.add_argument("--data.test_shot", type=int, default=0, metavar="TESTSHOT",
                    help="number of support examples per class in test. 0 means same as data.shot (default: 0)")
parser.add_argument("--data.test_query", type=int, default=15, metavar="TESTQUERY",
                    help="number of query examples per class in test. 0 means same as data.query (default: 15)")

parser.add_argument("--data.train_episodes", type=int, default=100, metavar="NTRAIN",
                    help="number of train episodes per epoch (default: 100)")
parser.add_argument("--data.test_episodes", type=int, default=100, metavar="NTEST",
                    help="number of test/val episodes per epoch (default: 100)")

parser.add_argument("--data.trainval", action="store_true",
                    help="run in train+validation mode (default: False)")
parser.add_argument("--data.sequential", action="store_true",
                    help="use sequential sampler instead of episodic (default: False)")
parser.add_argument("--data.cuda", action="store_true",
                    help="run in CUDA mode (default: False)")

# xBD dataset configuration
parser.add_argument("--data.root", type=str, default="", metavar="DATA.ROOT",
                    help="root of xBD-style dataset with train/val/test (required for xbd dataset)")
parser.add_argument("--data.patch_size", type=int, default=128, metavar="DATA.PATCH_SIZE",
                    help="patch size for xbd episodic sampling (default: 128)")
parser.add_argument("--data.min_class_pixels", type=int, default=32, metavar="DATA.MIN_CLASS_PIXELS",
                    help="minimum pixels of class inside patch to accept it (default: 32)")
parser.add_argument("--data.max_tries", type=int, default=80, metavar="DATA.MAX_TRIES",
                    help="max random attempts per sample to find a valid patch (default: 80)")
parser.add_argument("--data.episode_max_retries", type=int, default=200, metavar="DATA.EPISODE_MAX_RETRIES",
                    help="max retries to form a valid episode before failing (default: 200)")
parser.add_argument("--data.class_ids", type=str, default="0,1,2,3,4", metavar="DATA.CLASS_IDS",
                    help="comma-separated list of class ids to sample (default: 0,1,2,3,4)")
parser.add_argument("--data.use_two_stream", action="store_true",
                    help="if set, dataset returns pre/post streams for proto_seg (default: False)")


# Model args
default_model_name = "protonet_conv"
parser.add_argument("--model.model_name", type=str, default=default_model_name, metavar="MODELNAME",
                    help=f"model name (default: {default_model_name})")

parser.add_argument("--model.x_dim", type=str, default="1,28,28", metavar="XDIM",
                    help="dimensionality of input images (default: '1,28,28')")

parser.add_argument("--model.hid_dim", type=int, default=64, metavar="HIDDIM",
                    help="dimensionality of hidden layers (default: 64)")
parser.add_argument("--model.z_dim", type=int, default=64, metavar="ZDIM",
                    help="dimensionality of embedding (default: 64)")

# extra args used by proto_seg (kept optional for now 1/6)
parser.add_argument("--model.feat_dim", type=int, default=64, metavar="FEATDIM",
                    help="feature channels for proto_seg encoder (default: 64)")
parser.add_argument("--model.use_two_stream", action="store_true",
                    help="if set, model expects dict inputs with pre/post streams (default: False)")
parser.add_argument("--model.in_ch_single", type=int, default=3, metavar="INCH",
                    help="channels of a single stream image (default: 3)")


# Train args
parser.add_argument("--train.epochs", type=int, default=10000, metavar="NEPOCHS",
                    help="max epochs to train (default: 10000)")
parser.add_argument("--train.optim_method", type=str, default="Adam", metavar="OPTIM",
                    help="optimization method (default: Adam)")
parser.add_argument("--train.learning_rate", type=float, default=0.001, metavar="LR",
                    help="learning rate (default: 0.001)")
parser.add_argument("--train.decay_every", type=int, default=20, metavar="LRDECAY",
                    help="epochs after which to decay LR (default: 20)")
parser.add_argument("--train.weight_decay", type=float, default=0.0, metavar="WD",
                    help="weight decay (default: 0.0)")
parser.add_argument("--train.patience", type=int, default=200, metavar="PATIENCE",
                    help="epochs to wait without val improvement before stopping (default: 200)")

# Runtime Control
parser.add_argument("--train.val_every", type=int, default=1, metavar="VALEVERY",
                    help="run validation every N epochs (default: 1)")
parser.add_argument("--train.min_delta", type=float, default=0.0, metavar="MINDELTA",
                    help="minimum val-loss improvement required to reset patience (default: 0.0)")
parser.add_argument("--train.save_every", type=int, default=1, metavar="SAVEEVERY",
                    help="save a 'last' checkpoint every N epochs (default: 1)")
parser.add_argument("--train.resume_path", type=str, default="", metavar="RESUME",
                    help="path to a checkpoint dict to resume from (default: '')")
parser.add_argument("--train.no_val", action="store_true",
                    help="disable validation entirely (useful for smoke tests)")

# Log args
parser.add_argument("--log.fields", type=str, default="loss,acc", metavar="FIELDS",
                    help="fields to monitor during training (default: 'loss,acc')")
parser.add_argument("--log.exp_dir", type=str, default="results", metavar="EXP_DIR",
                    help="directory where experiments should be saved (default: 'results')")

args = vars(parser.parse_args())
main(args)
