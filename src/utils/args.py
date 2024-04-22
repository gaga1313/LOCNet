from argparse import ArgumentParser
from timm import utils


parser = ArgumentParser(description="Pytorch Imagenet Training")

parser.add_argument("--dataset", default="Imagenet")
parser.add_argument("--image_dir", help="Path to the image dataset")
parser.add_argument("--depth_dir", help="depth_maps")
parser.add_argument("--save-ddir", default=None, type=str, help="Save depth images dir")

parser.add_argument("--resume", default=False, type=bool)
parser.add_argument("--epochs", default=300, type=int)
parser.add_argument("--start-epoch", default=0, type=int)
parser.add_argument("--batch_size", type=int, default=128)


parser.add_argument(
    "--loss_alpha", default=1.0, type=float, help="Loss importance for classification"
)
parser.add_argument(
    "--loss_beta", default=1.0, type=float, help="scaling parameter for MSE loss"
)
parser.add_argument(
    "--rest-cce", default=1.0, type=int, help="freeze cce loss for initial rest epochs"
)

## model parameters
parser.add_argument("--model", help="name of the model", default="resnet50")
parser.add_argument("--pretrained", default=True) 
parser.add_argument("--initial_checkpoint", default=None)
parser.add_argument("--checkpoint-hist", default=10, type=int)
parser.add_argument("--grad-accum-steps", default=1)
parser.add_argument(
    "--smoothing", type=float, default=None, help="Label smoothing (default: None)"
)
parser.add_argument(
    "--output",
    default="checkpoint",
    type=str,
    metavar="PATH",
    help="path to output folder (default: none, current dir)",
)


parser.add_argument("--log_interval", default=100, type=int)

parser.add_argument(
    "--opt",
    default="sgd",
    type=str,
    metavar="OPTIMIZER",
    help='Optimizer (default: "sgd")',
)
parser.add_argument(
    "--opt-eps",
    default=None,
    type=float,
    metavar="EPSILON",
    help="Optimizer Epsilon (default: None, use opt default)",
)
parser.add_argument(
    "--opt-betas",
    default=None,
    type=float,
    nargs="+",
    metavar="BETA",
    help="Optimizer Betas (default: None, use opt default)",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.875,
    metavar="M",
    help="Optimizer momentum (default: 0.9)",
)
parser.add_argument(
    "--weight-decay", type=float, default=3.05e-5, help="weight decay (default: 2e-5)"
)
parser.add_argument(
    "--clip-grad",
    type=float,
    default=None,
    metavar="NORM",
    help="Clip gradient norm (default: None, no clipping)",
)
parser.add_argument(
    "--clip-mode",
    type=str,
    default="norm",
    help='Gradient clipping mode. One of ("norm", "value", "agc")',
)
parser.add_argument(
    "--layer-decay",
    type=float,
    default=None,
    help="layer-wise learning rate decay (default: None)",
)
parser.add_argument("--opt-kwargs", nargs="*", default={}, action=utils.ParseKwargs)

## Learning rate parameters
parser.add_argument(
    "--sched",
    type=str,
    default="cosine",
    metavar="SCHEDULER",
    help='LR scheduler (default: "step"',
)
parser.add_argument(
    "--sched-on-updates",
    action="store_true",
    default=False,
    help="Apply LR scheduler step on update instead of epoch end.",
)
parser.add_argument(
    "--lr",
    type=float,
    default=1.5e-3,
    metavar="LR",
    help="learning rate, overrides lr-base if set (default: None)",
)

parser.add_argument(
    "--warmup-lr",
    type=float,
    default=0.1,
    metavar="LR",
    help="warmup learning rate (default: 1e-5)",
)
parser.add_argument(
    "--min-lr",
    type=float,
    default=0,
    metavar="LR",
    help="lower lr bound for cyclic schedulers that hit 0 (default: 0)",
)

parser.add_argument(
    "--warmup-epochs",
    type=int,
    default=5,
    metavar="N",
    help="epochs to warmup LR, if scheduler supports",
)

parser.add_argument(
    "--patience-epochs",
    type=int,
    default=10,
    metavar="N",
    help="patience epochs for Plateau LR scheduler (default: 10)",
)

# parser.add_argument('Augmentation and regularization parameters')
parser.add_argument(
    "--aug",
    action="store_true",
    default=False,
    help="Enable all training augmentations",
)
parser.add_argument(
    "--no-aug",
    action="store_true",
    default=False,
    help="Disable all training augmentation, override other train aug args",
)
parser.add_argument(
    "--scale",
    type=float,
    nargs="+",
    default=[0.08, 1.0],
    metavar="PCT",
    help="Random resize scale (default: 0.08 1.0)",
)
parser.add_argument(
    "--ratio",
    type=float,
    nargs="+",
    default=[3.0 / 4.0, 4.0 / 3.0],
    metavar="RATIO",
    help="Random resize aspect ratio (default: 0.75 1.33)",
)
parser.add_argument(
    "--hflip", type=float, default=0.5, help="Horizontal flip training aug probability"
)
parser.add_argument(
    "--vflip", type=float, default=0.0, help="Vertical flip training aug probability"
)
parser.add_argument(
    "--color-jitter",
    type=float,
    default=0.4,
    metavar="PCT",
    help="Color jitter factor (default: 0.4)",
)
parser.add_argument(
    "--aa",
    type=str,
    default=None,
    metavar="NAME",
    help='Use AutoAugment policy. "v0" or "original". (default: None)',
),
parser.add_argument(
    "--aug-repeats",
    type=float,
    default=0,
    help="Number of augmentation repetitions (distributed training only) (default: 0)",
)
parser.add_argument(
    "--aug-splits",
    type=int,
    default=0,
    help="Number of augmentation splits (default: 0, valid: 0 or >=2)",
)
parser.add_argument(
    "--jsd-loss",
    action="store_true",
    default=False,
    help="Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.",
)
parser.add_argument(
    "--bce-loss",
    action="store_true",
    default=False,
    help="Enable BCE loss w/ Mixup/CutMix use.",
)
parser.add_argument(
    "--bce-target-thresh",
    type=float,
    default=None,
    help="Threshold for binarizing softened BCE targets (default: None, disabled)",
)
parser.add_argument(
    "--reprob",
    type=float,
    default=0.0,
    metavar="PCT",
    help="Random erase prob (default: 0.)",
)
parser.add_argument(
    "--remode", type=str, default="pixel", help='Random erase mode (default: "pixel")'
)
parser.add_argument(
    "--recount", type=int, default=1, help="Random erase count (default: 1)"
)
parser.add_argument(
    "--resplit",
    action="store_true",
    default=False,
    help="Do not random erase first (clean) augmentation split",
)
parser.add_argument(
    "--mixup",
    type=float,
    default=0.0,
    help="mixup alpha, mixup enabled if > 0. (default: 0.)",
)
parser.add_argument(
    "--cutmix",
    type=float,
    default=0.0,
    help="cutmix alpha, cutmix enabled if > 0. (default: 0.)",
)
parser.add_argument(
    "--cutmix-minmax",
    type=float,
    nargs="+",
    default=None,
    help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
)
parser.add_argument(
    "--mixup-prob",
    type=float,
    default=1.0,
    help="Probability of performing mixup or cutmix when either/both is enabled",
)
parser.add_argument(
    "--mixup-switch-prob",
    type=float,
    default=0.5,
    help="Probability of switching to cutmix when both mixup and cutmix enabled",
)
parser.add_argument(
    "--mixup-mode",
    type=str,
    default="batch",
    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
)
parser.add_argument(
    "--mixup-off-epoch",
    default=0,
    type=int,
    metavar="N",
    help="Turn off mixup after this epoch, disabled if 0 (default: 0)",
)

parser.add_argument(
    "--train-interpolation",
    type=str,
    default="random",
    help='Training interpolation (random, bilinear, bicubic default: "random")',
)

parser.add_argument("--rank", default=0, type=int)

parser.add_argument(
    "--no-prefetcher",
    action="store_true",
    default=False,
    help="disable fast prefetcher",
)

parser.add_argument(
    "-j",
    "--workers",
    type=int,
    default=1,
    metavar="N",
    help="how many training processes to use (default: 4)",
)

##Multiprocessing arguments
parser.add_argument(
    "--amp",
    action="store_true",
    default=False,
    help="use NVIDIA Apex AMP or Native AMP for mixed precision training",
)
parser.add_argument(
    "--amp-dtype",
    default="float16",
    type=str,
    help="lower precision AMP dtype (default: float16)",
)
parser.add_argument(
    "--amp-impl",
    default="native",
    type=str,
    help='AMP impl to use, "native" or "apex" (default: native)',
)
parser.add_argument(
    "--no-ddp-bb",
    action="store_true",
    default=False,
    help="Force broadcast buffers for native DDP to off.",
)
parser.add_argument(
    "--synchronize-step",
    action="store_true",
    default=False,
    help="torch.cuda.synchronize() end of each step",
)
parser.add_argument(
    "--pin-mem",
    action="store_true",
    default=False,
    help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
)

parser.add_argument(
    "--experiment",
    default="",
    type=str,
    metavar="NAME",
    help="name of train experiment, name of sub-folder for output",
)

parser.add_argument(
    "--log-wandb",
    action="store_true",
    default=False,
    help="log training and validation metrics to wandb",
)
parser.add_argument(
    "--log-dir", type=str, default="logs", help="directory to store logs"
)

args = parser.parse_args()
