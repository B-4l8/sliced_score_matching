import argparse
import logging
import os
import yaml
import torch
import numpy as np
from runners.dkef_runner import DKEFRunner, compute_sigma_list
from models.dkef import DKEF
from evaluations.hmc import HMCSampler
import sys
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    parser = argparse.ArgumentParser(description="Standalone HMC Sampling for DKEF")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--doc", type=str, required=True, help="Experiment name (folder in run/results)")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for sampling")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    with open(os.path.join("configs", args.config), "r") as f:
        config_dict = yaml.safe_load(f)
    config = dict2namespace(config_dict)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logging.info(f"Loading experiment: {args.doc}")

    runner_args = argparse.Namespace()
    runner_args.run = "run"
    runner_args.doc = args.doc
    runner_args.seed = args.seed
    runner_args.scalability_dim = config.data.input_dim
    runner_args.dsm_sigma = getattr(config.training, 'dsm_sigma', 0.1)

    runner = DKEFRunner(runner_args, config)

    train_data, val_data, test_data = runner.get_dataset()

    sigma_list = compute_sigma_list(train_data, config.model.num_kernels)

    if hasattr(config.model, 'M'):
        M = config.model.M
    else:
        M = 200

    init_z = torch.randn(M, config.data.input_dim)

    num_layers = getattr(config.model, 'num_layers', 3)

    dkef = DKEF(config.data.input_dim, mode=config.training.algo,
                num_kernels=config.model.num_kernels,
                init_z=init_z, hidden_dim=config.model.hidden_dim,
                num_layers=num_layers, add_skip=config.model.add_skip,
                alpha_param=config.model.alpha_param, train_Z=config.model.train_Z,
                pretrained_encoder=None, dsm_sigma=(runner_args.dsm_sigma if config.training.algo == 'dsm' else None), sigma_list=sigma_list).to(device)

    model_path = os.path.join("run", "results", args.doc, "model.pt")
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at {model_path}")
        return

    logging.info(f"Loading weights from {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    dkef.load_state_dict(state_dict)
    dkef.eval()

    if hasattr(dkef, 'alpha'):
        dkef.alpha = dkef.alpha.to(device)

    def energy_fn(x):
         return -dkef(None, x, stage="eval")

    hmc_sampler = HMCSampler(energy_fn, stepsize=1.0, n_steps=10)

    logging.info(f"Starting sampling: {args.samples} samples")

    total_acc_rate = 0
    generated_samples_list = []
    trace_list = []
    num_batches = (args.samples + args.batch_size - 1) // args.batch_size

    for i in range(num_batches):
        current_batch_size = min(args.batch_size, args.samples - i * args.batch_size)
        initial_samples = torch.randn(current_batch_size, config.data.input_dim).to(device)

        logging.info(f"Batch {i+1}/{num_batches}...")
        batch_samples, batch_trace = hmc_sampler.run_hmc_sampler(initial_samples, num_steps=100, return_trace=True)
        generated_samples_list.append(batch_samples.detach().cpu())
        trace_list.append(batch_trace)
        total_acc_rate += hmc_sampler.avg_acceptance_rate

    generated_samples = torch.cat(generated_samples_list, dim=0).numpy()

    logging.info("Processing trace for visualization...")

    full_trace = torch.cat(trace_list, dim=1)

    log_prob_trace = []
    chunk_size = 100

    with torch.no_grad():
        for t in range(full_trace.shape[0]):
            step_log_probs = []

            for i in range(0, full_trace.shape[1], chunk_size):
                batch_x = full_trace[t, i:i+chunk_size].to(device)

                batch_log_prob = -energy_fn(batch_x)
                step_log_probs.append(batch_log_prob.cpu())

            step_log_probs = torch.cat(step_log_probs)
            log_prob_trace.append(step_log_probs.mean().item())

    var_trace = full_trace[:, :5, 0].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(log_prob_trace)
    axes[0].set_title("Average Log Likelihood Trace")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Log Likelihood")
    axes[0].grid(True)

    axes[1].plot(var_trace)
    axes[1].set_title("Trace of First Dimension (First 5 Samples)")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Value")
    axes[1].grid(True)

    plt.tight_layout()
    plot_path = os.path.join("run", "results", args.doc, f"trace_plot_{args.samples}.png")
    plt.savefig(plot_path)
    logging.info(f"Saved trace plot to {plot_path}")
    plt.close()

    avg_acceptance_rate_final = total_acc_rate / num_batches
    logging.info("HMC Final Step Size: {:.6f}".format(hmc_sampler.stepsize))
    logging.info("HMC Average Acceptance Rate: {:.4f}".format(avg_acceptance_rate_final))

    logging.info("Inverse transforming samples...")
    scaler_path = os.path.join("run", "results", args.doc, "scaler.pkl")

    if os.path.exists(scaler_path):
        logging.info(f"Loading scaler from {scaler_path}")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        generated_samples = scaler.inverse_transform(generated_samples)
    else:
        logging.info("Scaler not found. Attempting fallback to runner's internal state.")
        scaler_type = getattr(config.data, 'scaler', 'pca_whiten')
        if scaler_type == 'pca_whiten':

            if hasattr(runner, 'Winv'):
                 generated_samples = runner.inv_whiten(generated_samples)
            else:
                 logging.warning("Inverse whitening matrix not found! Returning scaled samples.")
        else:
             generated_samples = runner.scaler.inverse_transform(generated_samples)

    out_path = os.path.join("run", "results", args.doc, f"{generated_samples.shape[1]}d_ksm.csv")
    np.savetxt(out_path, generated_samples, delimiter=",", header=",".join([f"dim_{i}" for i in range(generated_samples.shape[1])]), comments="")
    logging.info(f"Saved samples to {out_path}")

if __name__ == "__main__":
    main()
