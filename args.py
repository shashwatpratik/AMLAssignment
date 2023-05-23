import argparse


def argument_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--root", type=str, default="./datasets", help="root path to data directory"
    )
    parser.add_argument("-a", "--arch", type=str, default="M3T")
    parser.add_argument("--height", type=int, default=128, help="height of an image")
    parser.add_argument("--width", type=int, default=128, help="width of an image")
    parser.add_argument("--depth", type=int, default=128, help="depth of an image")
    parser.add_argument("--evaluate", action="store_true", help="evaluate only")
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=-1,
        help="evaluation frequency (set to -1 to test only in the end)",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adam",
        help="optimization algorithm (see optimizers.py)",
    )
    parser.add_argument(
        "--lr", default=0.0003, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--max-epoch", default=60, type=int, help="maximum epochs to run"
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        help="manual epoch number (useful when restart)",
    )
    parser.add_argument(
        "--start-eval",
        type=int,
        default=0,
        help="start to evaluate after a specific epoch",
    )

    parser.add_argument(
        "--train-batch-size", default=4, type=int, help="training batch size"
    )
    parser.add_argument(
        "--test-batch-size", default=2, type=int, help="test batch size"
    )
    parser.add_argument(
        "--no-pretrained", action="store_true", help="do not load pretrained weights"
    )
    parser.add_argument(
        "--load-weights",
        type=str,
        default="",
        help="load pretrained weights but ignore layers that don't match in size",
    )
    parser.add_argument("--use-cpu", action="store_true", help="use cpu")
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        help="number of data loading workers (tips: 4 or 8 times number of gpus)",
    )
    parser.add_argument("--seed", type=int, default=1, help="manual seed")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        metavar="PATH",
        help="resume from a checkpoint",
    )
    parser.add_argument(
        "--label-smooth",
        action="store_true",
        help="use label smoothing regularizer in cross entropy loss",
    )
    parser.add_argument(
        "--weight-decay", default=5e-04, type=float, help="weight decay"
    )
    # sgd
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        help="momentum factor for sgd and rmsprop",
    )
    parser.add_argument(
        "--sgd-dampening", default=0, type=float, help="sgd's dampening for momentum"
    )
    parser.add_argument(
        "--sgd-nesterov",
        action="store_true",
        help="whether to enable sgd's Nesterov momentum",
    )
    # rmsprop
    parser.add_argument(
        "--rmsprop-alpha", default=0.99, type=float, help="rmsprop's smoothing constant"
    )
    # adam/amsgrad
    parser.add_argument(
        "--adam-beta1",
        default=0.9,
        type=float,
        help="exponential decay rate for adam's first moment",
    )
    parser.add_argument(
        "--adam-beta2",
        default=0.999,
        type=float,
        help="exponential decay rate for adam's second moment",
    )
    parser.add_argument(
        "--gpu-devices",
        default="0",
        type=str,
        help="gpu device ids for CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument(
        "--use-avai-gpus",
        action="store_true",
        help="use available gpus instead of specified devices (useful when using managed clusters)",
    )
    parser.add_argument(
        "--save-dir", type=str, default="log", help="path to save log and model weights"
    )
    return parser



def optimizer_kwargs(parsed_args):
    """
    Build kwargs for optimizer in optimizers.py from
    the parsed command-line arguments.
    """
    return {
        "optim": parsed_args.optim,
        "lr": parsed_args.lr,
        "weight_decay": parsed_args.weight_decay,
        "momentum": parsed_args.momentum,
        "sgd_dampening": parsed_args.sgd_dampening,
        "sgd_nesterov": parsed_args.sgd_nesterov,
        "rmsprop_alpha": parsed_args.rmsprop_alpha,
        "adam_beta1": parsed_args.adam_beta1,
        "adam_beta2": parsed_args.adam_beta2,
    }