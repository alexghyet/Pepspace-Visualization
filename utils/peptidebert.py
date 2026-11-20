"""Shared PeptideBERT inference for the CLI and uploaded peptide sequences."""

import hashlib
from pathlib import Path

# Exact vocabulary from PeptideBERT/data/convert_encodings.py.
VOCAB = dict(zip(
    ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'L', 'A', 'G',
     'V', 'E', 'S', 'I', 'K', 'R', 'D', 'T', 'P', 'N', 'Q', 'F',
     'Y', 'M', 'H', 'C', 'W', 'X', 'U', 'B', 'Z', 'O'], range(30)))


def encode_sequence(sequence, vocab_size, max_length):
    if not isinstance(sequence, str) or not sequence.strip():
        raise ValueError('Every row must contain a nonempty peptide sequence.')
    sequence = sequence.strip().upper()
    if len(sequence) > max_length:
        raise ValueError(f'Sequence length {len(sequence)} exceeds {max_length}; no silent truncation.')
    if any(aa not in VOCAB for aa in sequence):
        raise ValueError('Sequence contains unsupported residues or internal whitespace.')
    ids = [VOCAB[aa] for aa in sequence]
    if max(ids) >= vocab_size:
        raise ValueError('Sequence contains residues outside this checkpoint vocabulary.')
    return ids


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def network_settings(config):
    """v1 stores settings directly; official v2 nests them under BERT."""
    network = config['network']
    return network.get('BERT', network)


def load_checkpoint(path):
    import numpy as np
    import torch

    # Official releases store NumPy scalar metrics alongside tensor weights.
    # Keep weights_only enabled; allow only these specific numeric types.
    with torch.serialization.safe_globals([
        (np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'),
        np.dtype, np.dtypes.Float64DType, np.dtypes.Float32DType,
    ]):
        return torch.load(path, map_location='cpu', weights_only=True)


def encoder_weights(checkpoint):
    if isinstance(checkpoint, dict) and 'bert_state_dict' in checkpoint:
        return checkpoint['bert_state_dict']
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        return {key.removeprefix('protbert.'): value
                for key, value in checkpoint['model_state_dict'].items()
                if key.startswith('protbert.')}
    raise ValueError('Expected a PeptideBERT model_state_dict or bert_state_dict checkpoint.')


def load_encoder(checkpoint_path, config_path, *, device="cpu", expected_hash=None, expected_config=None):
    import yaml
    from transformers import BertConfig, BertModel

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise ValueError("PeptideBERT weights are missing. Restore the official solubility checkpoint in .cache/peptidebert/v2.0/sol/.")
    if expected_hash and sha256(checkpoint_path) != expected_hash:
        raise ValueError("Local PeptideBERT weights do not match the Veltri reference checkpoint.")
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    network = network_settings(config)
    if expected_config is not None and (network != network_settings(expected_config)
                                        or config['vocab_size'] != expected_config['vocab_size']):
        raise ValueError("Local PeptideBERT configuration differs from the Veltri reference configuration.")
    bert_config = BertConfig(
        vocab_size=config['vocab_size'], hidden_size=network['hidden_size'],
        num_hidden_layers=network['hidden_layers'], num_attention_heads=network['attn_heads'],
        hidden_dropout_prob=network['dropout'],
    )
    model = BertModel(bert_config)
    model.load_state_dict(encoder_weights(load_checkpoint(checkpoint_path)), strict=True)
    return model.to(device).eval()


def embed_sequences(sequences, model, *, batch_size=16, progress=None):
    import numpy as np
    import torch

    from utils.dataset import validate_embeddings

    if not len(sequences) or batch_size < 1:
        raise ValueError("Provide at least one sequence and a positive batch size.")
    encoded = []
    for row, sequence in enumerate(sequences):
        try:
            encoded.append(encode_sequence(sequence, model.config.vocab_size, model.config.max_position_embeddings))
        except ValueError as error:
            raise ValueError(f"CSV row {row + 2}: {error}") from error
    device = next(model.parameters()).device
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            batch = encoded[start:start + batch_size]
            ids = torch.zeros((len(batch), max(map(len, batch))), dtype=torch.long, device=device)
            for index, tokens in enumerate(batch):
                ids[index, :len(tokens)] = torch.tensor(tokens, device=device)
            mask = ids.ne(0)
            hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
            pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
            vectors.append(pooled.cpu().numpy().astype("float32"))
            if progress:
                progress(min(start + batch_size, len(encoded)), len(encoded))
    return validate_embeddings(np.vstack(vectors), min_rows=1)

