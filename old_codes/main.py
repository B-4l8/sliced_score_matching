import argparse
import traceback
import time
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
from runners.dkef_runner import DKEFRunner

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--runner', type=str, default='DKEFRunner', help='The runner to execute')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--run', type=str, default='run', help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, required=True, help='A string for documentation purpose')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--test', action='store_true', help='Whether to test the model')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('--dsm_sigma', type=float, default=0.16, help='Sigma for DSM tuning')
    
    args = parser.parse_args()
    
    args.log = os.path.join(args.run, 'logs', args.doc)

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    if not os.path.exists(args.log):
        os.makedirs(args.log)

    # setup logger
    level = getattr(logging, args.verbose.upper(), None)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    
    # Check if root logger has handlers, clear them if so to avoid duplication
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
        
    handler1 = logging.StreamHandler()
    handler1.setFormatter(formatter)
    handler2 = logging.FileHandler(os.path.join(args.log, 'stdout.txt'))
    handler2.setFormatter(formatter)
    
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info(f"Starting Data Generation Task with doc: {args.doc}...")
    
    try:
        runner = DKEFRunner(args, config)
        runner.train()
        
        # Post-processing: Convert npy to CSV
        output_npy = os.path.join(args.run, 'results', args.doc, 'generated_samples.npy')
        # Dynamic output filename based on doc arg
        output_csv = os.path.join(args.run, 'results', args.doc, f'generated_{args.doc}.csv')
        
        if os.path.exists(output_npy):
            data = np.load(output_npy)
            # Create header V1, V2, ...
            header = ",".join([f'V{i+1}' for i in range(data.shape[1])])
            np.savetxt(output_csv, data, delimiter=",", header=header, comments="")
            logging.info(f"Successfully saved generated samples to {output_csv}")
        else:
            logging.error("generated_samples.npy not found!")
            
    except Exception as e:
        logging.error(traceback.format_exc())
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
