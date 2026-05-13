from pathlib import Path
import sys
import itertools
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.clustering import cluster_kmeans
from src.evaluation import evaluate_clustering

# Feature files
MODALITIES = {
    'EEG': 'results/mesa_eeg_features_aligned.npz',
    'EOG': 'results/mesa_eog_features_aligned.npz',
    'EMG': 'results/mesa_emg_features_aligned.npz',
    'HRV': 'results/mesa_features.npz',
}

# Load all feature sets
loaded = {}
for name, path in MODALITIES.items():
    data = np.load(PROJECT_ROOT / path, allow_pickle=True)
    loaded[name] = {
        'X': data['X'],
        'y': data['y'],
    }
    print(f'{name}: X={data["X"].shape}, y={data["y"].shape}')

# Use common number of epochs
min_epochs = min(len(v['X']) for v in loaded.values())
y_ref = loaded['EEG']['y'][:min_epochs]

# Fixed random subset for fair comparison
np.random.seed(42)
idx = np.random.choice(
    min_epochs,
    size=min(10000, min_epochs),
    replace=False
)

results = []

# Test every non-empty combination
for r in range(1, len(MODALITIES) + 1):
    for combo in itertools.combinations(MODALITIES.keys(), r):
        combo_name = ' + '.join(combo)
        print(f'\nEvaluating: {combo_name}')

        # Concatenate features
        X_parts = [loaded[m]['X'][:min_epochs] for m in combo]
        X = np.hstack(X_parts)
        y = y_ref

        # Impute and standardize
        X = SimpleImputer(strategy='median').fit_transform(X)
        X = StandardScaler().fit_transform(X)

        # Same 10,000 epochs for every experiment
        X_sub = X[idx]
        y_sub = y[idx]

        # Cluster
        labels = cluster_kmeans(X_sub, n_clusters=5)

        # Evaluate
        metrics = evaluate_clustering(
            X_sub,
            labels,
            ground_truth=y_sub
        )

        results.append({
            'combination': combo_name,
            'n_features': X.shape[1],
            'ari': metrics['ari'],
            'nmi': metrics['nmi'],
            'ami': metrics['ami'],
            'f_score': metrics['f_score'],
            'silhouette': metrics['silhouette'],
        })

# Save results
df = pd.DataFrame(results)
df_sorted = df.sort_values('ari', ascending=False)

results_dir = PROJECT_ROOT / 'results'
results_dir.mkdir(exist_ok=True)

csv_path = results_dir / 'exhaustive_ablation_results.csv'
df_sorted.to_csv(csv_path, index=False)

print('\n' + '=' * 80)
print('ALL COMBINATIONS SORTED BY ARI')
print('=' * 80)
print(df_sorted.to_string(index=False))

print(f'\nSaved results to: {csv_path}')

print('\nBEST BY ARI:')
print(df_sorted.iloc[0][['combination', 'ari', 'nmi', 'f_score']])

print('\nBEST BY NMI:')
print(df.loc[df['nmi'].idxmax()][['combination', 'ari', 'nmi', 'f_score']])

print('\nBEST BY F1:')
print(df.loc[df['f_score'].idxmax()][['combination', 'ari', 'nmi', 'f_score']])
