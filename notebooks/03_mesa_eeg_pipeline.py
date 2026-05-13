from pathlib import Path
import sys
import numpy as np
import mne
from tqdm import tqdm

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MESA_EDF_DIR, MESA_XML_DIR, EPOCH_LENGTH_SEC
from src.eeg_features import extract_eeg_features_all_epochs
from src.preprocessing import load_aasm_annotations, expand_stages_to_epochs

# --------------------------------------------------
# 1. Select the same 100-subject subset as ECG
# --------------------------------------------------
edf_files = sorted(MESA_EDF_DIR.glob('*.edf'))

np.random.seed(42)
selected_files = np.random.choice(
    edf_files,
    size=min(100, len(edf_files)),
    replace=False,
)

print(f'Selected {len(selected_files)} subjects')

# --------------------------------------------------
# 2. Process all subjects
# --------------------------------------------------
all_features = []
all_labels = []
failed = []

for edf_path in tqdm(selected_files, desc='Processing EEG'):
    try:
        # Corresponding XML annotation file
        xml_path = MESA_XML_DIR / f"{edf_path.stem}-nsrr.xml"
        if not xml_path.exists():
            failed.append((edf_path.name, 'Missing XML annotation'))
            continue

        # Load sleep stage labels
        stage_events = load_aasm_annotations(xml_path)
        _, epoch_labels = expand_stages_to_epochs(stage_events)

        if len(epoch_labels) == 0:
            failed.append((edf_path.name, 'No stage labels found'))
            continue

        # Load only EEG channels
        raw = mne.io.read_raw_edf(
            str(edf_path),
            include=['EEG1', 'EEG2', 'EEG3'],
            preload=True,
            verbose=False,
        )

        data = raw.get_data()  # shape: (3, n_samples)
        sr = raw.info['sfreq']

        # Segment into 30-second epochs
        samples_per_epoch = int(EPOCH_LENGTH_SEC * sr)
        n_signal_epochs = data.shape[1] // samples_per_epoch

        # Keep only epochs present in BOTH signal and annotations
        n_epochs = min(n_signal_epochs, len(epoch_labels))

        if n_epochs == 0:
            failed.append((edf_path.name, 'No overlapping epochs'))
            continue

        # Trim signal and labels to same length
        data = data[:, :n_epochs * samples_per_epoch]
        labels = epoch_labels[:n_epochs]

        # Reshape to (n_epochs, 3, samples_per_epoch)
        epochs = data.reshape(
            3,
            n_epochs,
            samples_per_epoch,
        ).transpose(1, 0, 2)

        # Extract EEG features
        X = extract_eeg_features_all_epochs(epochs, sr)

        # Store
        all_features.append(X)
        all_labels.append(labels)

    except Exception as e:
        failed.append((edf_path.name, str(e)))

# --------------------------------------------------
# 3. Concatenate
# --------------------------------------------------
if not all_features:
    raise RuntimeError('No EEG features were extracted.')

X = np.vstack(all_features)
y = np.concatenate(all_labels)

# --------------------------------------------------
# 4. Save
# --------------------------------------------------
results_dir = PROJECT_ROOT / 'results'
results_dir.mkdir(exist_ok=True)

output_path = results_dir / 'mesa_eeg_features_aligned.npz'
np.savez(output_path, X=X, y=y)

print('\nSaved aligned EEG features to:', output_path)
print('Feature matrix shape:', X.shape)
print('Labels shape:', y.shape)
print('Features per epoch:', X.shape[1])
print('Failed subjects:', len(failed))

if failed:
    print('\nFirst 5 failures:')
    for name, msg in failed[:5]:
        print(f'  {name}: {msg}')
