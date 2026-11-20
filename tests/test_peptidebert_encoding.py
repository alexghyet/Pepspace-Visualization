import unittest

from utils.peptidebert import encode_sequence, network_settings, encoder_weights


class PeptideBERTEncodingTests(unittest.TestCase):
    def test_both_checkpoint_layouts(self):
        weights = {'embeddings.word_embeddings.weight': 'test sentinel'}
        self.assertEqual(encoder_weights({'bert_state_dict': weights}), weights)
        self.assertEqual(encoder_weights({'model_state_dict': {
            'protbert.embeddings.word_embeddings.weight': 'test sentinel',
            'head.0.weight': 'excluded',
        }}), weights)
        with self.assertRaises(ValueError):
            encoder_weights({})

    def test_both_release_config_layouts(self):
        settings = {'hidden_size': 256, 'hidden_layers': 8, 'attn_heads': 8, 'dropout': 0.1}
        self.assertEqual(network_settings({'network': settings}), settings)
        self.assertEqual(network_settings({'network': {'BERT': settings}}), settings)

    def test_matches_upstream_residue_ids_without_special_tokens(self):
        self.assertEqual(encode_sequence(' lagvcw ', 25, 512), [5, 6, 7, 8, 23, 24])

    def test_rejects_unsupported_checkpoint_residues(self):
        for sequence in ('AX', 'AU', 'A B', '', None, 'A*'):
            with self.subTest(sequence=sequence), self.assertRaises(ValueError):
                encode_sequence(sequence, 25, 512)

    def test_rejects_truncation(self):
        with self.assertRaises(ValueError):
            encode_sequence('AAAA', 25, 3)


if __name__ == '__main__':
    unittest.main()
