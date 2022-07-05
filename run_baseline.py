import argparse

import run_colof
from lve.colof import FULL_EXP_TYPE, default_params
from pathlib import Path

from settings import smurf_weights_path, flownets_weights_path, raft_weights_path


def create_comp_exp(properties):
    args = argparse.Namespace(**properties)
    args.freeze = 'yes'
    args.exp_type = FULL_EXP_TYPE
    args.net_flow_input_type = 'implicit'

    if args.arch == 'sota-smurf':
        args.load = smurf_weights_path
    elif args.arch == 'sota-raft' or args.arch == 'sota-raft-small':
        args.load = raft_weights_path
    elif args.arch == 'sota-flownets':
        args.load = flownets_weights_path

    if args.arch == 'none-ihs':
        args.force_gray = 'yes'
        args.net_flow_input_type = 'explicit'
        args.step_size = 0
        args.weight_decay = 0

    # adding any default parameters if not already specified
    args_dict = vars(args)
    for (k, v) in default_params.items():
        if k not in args_dict: args_dict[k] = v

    print('-- launching run', args)
    run_colof.run_exp(args)

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--arch', type=str, default="none",
                        choices=["none", "none-ihs", "sota-flownets", "sota-smurf", "sota-raft", "sota-raft-small"])
    parser.add_argument('--load', type=str, help='path to pretrained model')
    parser.add_argument('--experience', type=str, default="a", choices=["a", "b", "c", "movie"])
    parser.add_argument('--iter_ihs', type=int, default=0, help='number of HS iterations (should be 0 unless the chosen arch is none-ihs')
    parser.add_argument('--warm_ihs', type=str, default=default_params['warm_ihs'], help='whether HS algorithm should be initialized with warm start')
    parser.add_argument('--lambda_s', type=float, default=default_params['lambda_s'], help='weight of the smoothness regularization')

    args_cmd = parser.parse_args()
    args_dict = vars(args_cmd)
    create_comp_exp(args_dict)

if __name__ == '__main__':
    main()
