from pathlib import Path
import sys
import numpy as np
import mne
from tqdm import tqdm

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MESA_EDF_DIR, MESA_XML_DIR, EPOCH_LENGTH_SEC
from src.preprocessing import load_aasm_annotations, expand_stages_to_epochs
from src.eog_features import extract_eog_features_all_epochs

# Same 100-subject subset
edf_files = sorted(MESA_EDF_DIR.glob('*.edf'))
np.random.seed(42)
selected_files = np.random.choice(
    edf_files,
    size=min(100, len(edf_files)),
    replace=False
)

print(f'Selected {len(selected_files)} subjects')

all_features = []
all_labels = []
failed = []

for edf_path in tqdm(selected_files, desc='Processing EOG'):
    try:
        xml_path = MESA_XML_DIR / f'{edf_path.stem}-nsrr.xml'
        if not xml_path.exists():
            continue

        # Load labels
        stage_events = load_aasm_annotations(xml_path)
        _, epoch_labels = expand_stages_to_epochs(stage_events)

        if len(epoch_labels) == 0:
            continue

        # Load only EOG channels
        raw = mne.io.read_raw_edf(
            str(edf_path),
            include=['EOG-L', 'EOG-R'],
            preload=True,
            verbose=False
        )

        data = raw.get_data()   # shape: (2, n_samples)
        sr = raw.info['sfreq']

        samples_per_epoch = int(EPOCH_LENGTH_SEC * sr)
        n_signal_epochs = data.shape[1] // samples_per_epoch
        n_epochs = min(n_signal_epochs, len(epoch_labels))

        if n_epochs == 0:
            continue

        # Trim to aligned length
        data = data[:, :n_epochs * samples_per_epoch]
        labels = epoch_labels[:n_epochs]

        # Reshape to (n_epochs, 2, samples_per_epoch)
        epochs = data.reshape(
            2,
            n_epochs,
            samples_per_epoch
        ).transpose(1, 0, 2)

        # Extract features
        X = extract_eog_features_all_epochs(epochs, sr)

        all_features.append(X)
        all_labels.append(labels)

    except Exception as e:
        failed.append((edf_path.name, str(e)))

# Combine all subjects
X = np.vstack(all_features)
y = np.concatenate(all_labels)

# Save results
results_dir = PROJECT_ROOT / 'results'
results_dir.mkdir(exist_ok=True)

output_path = results_dir / 'mesa_eog_features_aligned.npz'
np.savez(output_path, X=X, y=y)

print('\nSaved:', output_path)
print('X shape:', X.shape)
print('y shape:', y.shape)
print('Features per epoch:', X.shape[1])
print('Failed subjects:', len(failed))
