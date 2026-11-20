"""Extract Veltri embeddings from a trained ChakradharG/PeptideBERT checkpoint.

Matches the repository's residue IDs and absence of CLS/SEP tokens. Does not
download base weights or silently substitute a randomly initialized encoder.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from utils.project_paths import EMBEDDING_DIRECTORY, VELTRI_PATH

from utils.peptidebert import (
    sha256, load_encoder, embed_sequences,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Trained model.pt from PeptideBERT')
    parser.add_argument('--config', type=Path, help='Matching config.yaml; defaults beside checkpoint')
    parser.add_argument('--csv', type=Path, default=BASE / VELTRI_PATH)
    parser.add_argument('--output-dir', type=Path, default=BASE / EMBEDDING_DIRECTORY)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None)
    args = parser.parse_args()
    config_path = args.config or args.checkpoint.parent / 'config.yaml'
    if not args.checkpoint.is_file():
        parser.error(f'Trained checkpoint not found: {args.checkpoint}. A repository clone is not a checkpoint.')
    if not config_path.is_file():
        parser.error(f'Matching configuration not found: {config_path}')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    output_paths = [args.output_dir / name for name in
                    ('veltri_embeddings.npy', 'veltri_metadata.csv', 'manifest.json')]
    if any(path.exists() for path in output_paths):
        parser.error('Output already exists; choose a new --output-dir to preserve previous results.')

    import numpy as np
    import pandas as pd
    import torch
    import yaml
    from utils.dataset import parse_amp_labels

    config = yaml.safe_load(config_path.read_text())
    frame = pd.read_csv(args.csv)
    if frame.empty or not {'aa_seq', 'AMP'}.issubset(frame.columns):
        raise ValueError('Expected a nonempty Veltri CSV with aa_seq and AMP columns.')
    labels = parse_amp_labels(frame['AMP'])
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_encoder(args.checkpoint, config_path, device=device)
    embeddings = embed_sequences(
        frame['aa_seq'].tolist(), model, batch_size=args.batch_size,
        progress=lambda done, total: print(f'Embedded {done}/{total}', flush=True),
    )
    metadata = pd.DataFrame({
        'source_row_index': np.arange(len(frame)), 'sequence': frame['aa_seq'].to_numpy(),
        'is_amp': np.asarray(labels, dtype=bool),
    })
    manifest = {
        'model': 'ChakradharG/PeptideBERT', 'task': config.get('task'),
        'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': sha256(args.checkpoint),
        'config': config, 'dataset_sha256': sha256(args.csv),
        'pooling': 'last_hidden_state masked residue mean',
        'encoding': 'PeptideBERT convert_encodings.m2; uppercase; no CLS/SEP; padding ID 0',
        'truncation': False, 'shape': list(embeddings.shape), 'device': device,
        'torch_version': torch.__version__,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_paths[0], embeddings)
    metadata.to_csv(output_paths[1], index=False)
    manifest['embeddings_sha256'] = sha256(output_paths[0])
    output_paths[2].write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Saved {embeddings.shape} embeddings and matching metadata to {args.output_dir}')


if __name__ == '__main__':
    main()
