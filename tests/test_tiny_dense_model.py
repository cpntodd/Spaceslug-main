import unittest

from spaceslug.tiny_dense_model import TinyDenseCausalModel


class TinyDenseCausalModelTest(unittest.TestCase):
    def test_analytic_gradient_matches_centered_difference(self):
        model = TinyDenseCausalModel.create(5, hidden_size=3)
        sequences = [[0, 1, 2], [2, 3, 1]]
        _, gradients = model.loss_and_gradients(sequences)
        epsilon = 1e-5
        cases = (("embedding", 1, 2), ("output", 2, 4), ("output_bias", 3, None))
        for name, first, second in cases:
            original = model.parameter(name, first, second)
            model.set_parameter(name, first, original + epsilon, second)
            plus, _ = model.loss_and_gradients(sequences)
            model.set_parameter(name, first, original - epsilon, second)
            minus, _ = model.loss_and_gradients(sequences)
            model.set_parameter(name, first, original, second)
            numerical = (plus - minus) / (2.0 * epsilon)
            analytic = gradients[name][first] if second is None else gradients[name][first][second]
            self.assertAlmostEqual(analytic, numerical, delta=1e-7)

    def test_fixed_batch_training_reduces_loss(self):
        model = TinyDenseCausalModel.create(4, hidden_size=4)
        sequences = [[0, 1, 0, 1], [0, 1, 0, 1]]
        before, _ = model.loss_and_gradients(sequences)
        for _ in range(100):
            model.train_step(sequences, 0.5)
        after, _ = model.loss_and_gradients(sequences)
        self.assertLess(after, before)


if __name__ == "__main__":
    unittest.main()
