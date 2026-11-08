"""Guard participant isolation and six-class checkpoint compatibility."""
import io
import unittest

import numpy as np
import torch

from biomat.classifier import MovementCNN
from biomat.phone_benchmark import make_folds, score


class PhoneBenchmarkTests(unittest.TestCase):
    def test_subject_folds_keep_repeated_windows_together(self):
        subjects = np.repeat(np.arange(21), np.arange(1, 22))
        held_out = []
        for fold in make_folds(subjects):
            train = set(fold['train_subjects'])
            validation = set(fold['validation_subjects'])
            self.assertFalse(train & validation)
            self.assertEqual(train | validation, set(subjects))
            held_out.extend(validation)
            for person in np.unique(subjects):
                mask = np.isin(subjects[subjects == person], fold['validation_subjects'])
                self.assertTrue(mask.all() or not mask.any())
        self.assertEqual(sorted(held_out), list(range(21)))

    def test_phone_checkpoint_roundtrip_and_six_class_metrics(self):
        model = MovementCNN(input_channels=6, num_classes=6).eval()
        x = torch.randn(6, 128, 6)
        with torch.inference_mode():
            expected = model(x)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = MovementCNN(6, 6).eval()
        restored.load_state_dict(torch.load(stream, weights_only=True))
        with torch.inference_mode():
            torch.testing.assert_close(restored(x), expected)
        result = score(np.arange(6), np.eye(6), np.ones(6, dtype=int))
        self.assertEqual(result['accuracy'], 1)
        self.assertEqual(result['macro_f1'], 1)
        self.assertEqual(np.asarray(result['confusion_matrix']).shape, (6, 6))


if __name__ == '__main__':
    unittest.main()
