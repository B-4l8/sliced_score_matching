import os
import numpy as np
import pandas as pd
import argparse
from sklearn.datasets import make_classification

def generate_data():
    parser = argparse.ArgumentParser(description='Generate synthetic classification data')
    parser.add_argument('--dim', type=int, default=500, help='Number of features (dimensions)')
    parser.add_argument('--n_neg', type=int, default=5000, help='Number of negative samples (class 0)')
    parser.add_argument('--n_pos', type=int, default=5000, help='Number of positive samples (class 1)')
    
    args = parser.parse_args()
    
    target_total = args.n_neg + args.n_pos
    buffer_multiplier = 2.0
    n_samples_gen = int(target_total * buffer_multiplier)
    
    pos_ratio = args.n_pos / target_total
    neg_ratio = args.n_neg / target_total
    
    print(f"Generating pool of {n_samples_gen} samples to extract {args.n_neg} neg and {args.n_pos} pos...")

    X, y = make_classification(
        n_samples=n_samples_gen,
        n_features=args.dim,
        n_informative=int(args.dim * 0.1),  # 차원의 10%
        n_redundant=int(args.dim * 0.1),    # 차원의 10%               
        n_repeated=int(args.dim * 0.1),    # 차원의 5%             
        n_classes=2,                        
        n_clusters_per_class=int(args.dim * 0.1),            # 기본 16
        weights=[neg_ratio, pos_ratio],     
        flip_y=0.01,                         
        class_sep=1.0,                         
        hypercube=True,
        shift=None,
        scale=1.0,
        shuffle=True,
        random_state=418
    )

    df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(args.dim)])
    df['target'] = y
    
    df_neg = df[df['target'] == 0]
    df_pos = df[df['target'] == 1]
    
    if len(df_neg) < args.n_neg:
        raise ValueError(f"Generated only {len(df_neg)} negative samples, wanted {args.n_neg}. Increase buffer or adjust weights.")
    if len(df_pos) < args.n_pos:
        raise ValueError(f"Generated only {len(df_pos)} positive samples, wanted {args.n_pos}. Increase buffer or adjust weights.")
        
    df_neg_sampled = df_neg.sample(n=args.n_neg, random_state=42)
    df_pos_sampled = df_pos.sample(n=args.n_pos, random_state=42)
    
    df_final = pd.concat([df_neg_sampled, df_pos_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("Final class counts:")
    print(df_final['target'].value_counts())

    output_dir = f'data/{args.dim}d'
    os.makedirs(output_dir, exist_ok=True)
    
    df_final.to_csv(f'{output_dir}/synthetic_full.csv', index=False)
    print(f"Saved full dataset to {output_dir}/synthetic_full.csv with shape {df_final.shape}")
    
    pos_data = df_pos_sampled.drop(columns=['target'])
    
    pos_path = os.path.join(output_dir, 'synthetic_pos.csv')
    pos_data.to_csv(pos_path, index=False)
    print(f"Saved positive class data to {pos_path} with shape {pos_data.shape}")

if __name__ == "__main__":
    generate_data()
