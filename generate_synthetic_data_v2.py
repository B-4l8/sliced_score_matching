import os
import numpy as np
import pandas as pd
import argparse
from sklearn.datasets import make_classification

def get_subset_indices(total_n, subset_n, current_indices):
    needed = subset_n - len(current_indices)
    if needed < 0:
        raise ValueError(f"subset_n ({subset_n}) is smaller than existing indices ({len(current_indices)})")
    return list(range(subset_n))

def generate_data():
    parser = argparse.ArgumentParser(description='Generate hierarchical synthetic classification data')
    parser.add_argument('--n_neg', type=int, default=10000, help='Number of negative samples (class 0)')
    parser.add_argument('--n_pos', type=int, default=10000, help='Number of positive samples (class 1)')

    args = parser.parse_args()

    dims = [25, 50, 100, 300, 500, 1000]
    base_dim = 1000

    r_inf = 0.04
    r_red = 0.4
    r_rep = 0.4

    n_inf_base = int(base_dim * r_inf)
    n_red_base = int(base_dim * r_red)
    n_rep_base = int(base_dim * r_rep)
    n_noise_base = base_dim - n_inf_base - n_red_base - n_rep_base
    n_clu_base = int(base_dim * 0.5)

    target_total = args.n_neg + args.n_pos
    buffer_multiplier = 1.2
    n_samples_gen = int(target_total * buffer_multiplier)

    pos_ratio = args.n_pos / target_total
    neg_ratio = args.n_neg / target_total

    print(f"Generating base 1000D dataset ({n_samples_gen} samples)...")

    X_base, y = make_classification(
        n_samples=n_samples_gen,
        n_features=base_dim,
        n_informative=n_inf_base,
        n_redundant=n_red_base,
        n_repeated=n_rep_base,
        n_classes=2,
        n_clusters_per_class=n_clu_base,
        weights=[neg_ratio, pos_ratio],
        flip_y=0.2,
        class_sep=1.0,
        hypercube=True,
        shift=True,
        scale=1.0,
        shuffle=False,
        random_state=418
    )

    idx_start_inf = 0
    idx_start_red = n_inf_base
    idx_start_rep = n_inf_base + n_red_base
    idx_start_rnd = n_inf_base + n_red_base + n_rep_base

    for dim in dims:
        print(f"Processing dimension {dim}...")

        n_inf = int(dim * r_inf)
        n_red = int(dim * r_red)
        n_rep = int(dim * r_rep)
        n_rnd = dim - n_inf - n_red - n_rep

        indices = []
        indices.extend(range(idx_start_inf, idx_start_inf + n_inf))
        indices.extend(range(idx_start_red, idx_start_red + n_red))
        indices.extend(range(idx_start_rep, idx_start_rep + n_rep))
        indices.extend(range(idx_start_rnd, idx_start_rnd + n_rnd))

        indices = np.array(indices)

        X_subset = X_base[:, indices]

        print(f"  > Counts: Inf={n_inf}, Red={n_red}, Rep={n_rep}, Rnd={n_rnd}")

        np.random.seed(dim)
        perm = np.random.permutation(X_subset.shape[1])
        X_subset_shuffled = X_subset[:, perm]

        df = pd.DataFrame(X_subset_shuffled, columns=[f'feat_{i}' for i in range(dim)])
        df['target'] = y

        df_neg = df[df['target'] == 0]
        df_pos = df[df['target'] == 1]

        if len(df_neg) < args.n_neg or len(df_pos) < args.n_pos:
             print(f"Warning: Not enough samples generated. Neg: {len(df_neg)}, Pos: {len(df_pos)}")

        df_neg_sampled = df_neg.sample(n=args.n_neg, random_state=42)
        df_pos_sampled = df_pos.sample(n=args.n_pos, random_state=42)

        df_final = pd.concat([df_neg_sampled, df_pos_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

        output_dir = f'data/{dim}d'
        os.makedirs(output_dir, exist_ok=True)

        full_path = f'{output_dir}/synthetic_full.csv'
        df_final.to_csv(full_path, index=False)

        pos_data = df_pos_sampled.drop(columns=['target'])
        pos_path = f'{output_dir}/synthetic_pos.csv'

        print(f"  > Saved to {output_dir}")

def split_data():
    dim = [25, 50, 100, 300, 500, 1000]
    base_path = "./data/"

    for d in dim:
        df = pd.read_csv(f"{base_path}/{d}d/synthetic_full.csv")
        df_train = df.iloc[:int(len(df)*0.5)]
        df_test = df.iloc[int(len(df)*0.5):]
        df_train_pos = df_train[df_train['target'] == 1].drop(columns=['target'])
        df_train_pos.to_csv(f"{base_path}/{d}d/synthetic_pos.csv", index=False)
        df_test.to_csv(f"{base_path}/{d}d/synthetic_test.csv", index=False)
        print(f"Saved {d}d")

if __name__ == "__main__":
    generate_data()
    split_data()
