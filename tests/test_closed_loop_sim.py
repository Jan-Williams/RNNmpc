import unittest
import numpy as np
from utils import closed_loop_sim 

class TestClosedLoopSim(unittest.TestCase):

    def setUp(self):
        """Set up shared resources for tests."""
        self.environment = "LorenzControl"
        self.reference_trajectory = np.zeros((3, 500))
        self.train_data = {
            "U_train": np.random.rand(1, 500),
            "S_train": np.random.rand(3, 500),
            "O_train": np.random.rand(3, 500)
        }
        self.control_params = {
            "dev": 100,
            "u_1": 1,
            "u_2": 20,
            "control_horizon": 20,
            "forecast_horizon": 50
        }
        self.hyperparams_lstm = {
            "Nr": 1000,
            "dropout_p": 0.2,
            "lags": 10,
            "adam_lr": 0.001
        }
        self.hyperparams_linear = {
            "tds": [-5, -4, -3, -2, -1],
            "beta": 0.01
        }

    def test_lstm_forecaster(self):
        """Test the function with LSTMForecaster."""
        results = closed_loop_sim(
            environment=self.environment,
            reference_trajectory=self.reference_trajectory,
            model_type="LSTMForecaster",
            train_data=self.train_data,
            hyperparams=self.hyperparams_lstm,
            control_params=self.control_params
        )
        self.assertIn("controlled_states", results)
        self.assertIn("applied_controls", results)
        self.assertIn("reference_trajectory", results)
        self.assertEqual(results["controlled_states"].shape[0], 3)  # Check dimensions
        self.assertEqual(results["applied_controls"].shape[0], 1)

    def test_linear_forecaster(self):
        """Test the function with LinearForecaster."""
        results = closed_loop_sim(
            environment=self.environment,
            reference_trajectory=self.reference_trajectory,
            model_type="LinearForecaster",
            train_data=self.train_data,
            hyperparams=self.hyperparams_linear,
            control_params=self.control_params
        )
        self.assertIn("controlled_states", results)
        self.assertIn("applied_controls", results)
        self.assertIn("reference_trajectory", results)
        self.assertEqual(results["controlled_states"].shape[0], 3)  # Check dimensions
        self.assertEqual(results["applied_controls"].shape[0], 1)

    def test_invalid_environment(self):
        """Test invalid environment raises a ValueError."""
        with self.assertRaises(ValueError):
            closed_loop_sim(
                environment="InvalidEnvironment",
                reference_trajectory=self.reference_trajectory,
                model_type="LSTMForecaster",
                train_data=self.train_data,
                hyperparams=self.hyperparams_lstm,
                control_params=self.control_params
            )

    def test_invalid_model_type(self):
        """Test invalid model type raises a ValueError."""
        with self.assertRaises(ValueError):
            closed_loop_sim(
                environment=self.environment,
                reference_trajectory=self.reference_trajectory,
                model_type="InvalidForecaster",
                train_data=self.train_data,
                hyperparams=self.hyperparams_lstm,
                control_params=self.control_params
            )

    def test_missing_train_data_key(self):
        """Test missing key in train_data raises a ValueError."""
        incomplete_train_data = {
            "U_train": np.random.rand(1, 500),
            "S_train": np.random.rand(3, 500)
        }  # Missing 'O_train'
        with self.assertRaises(ValueError):
            closed_loop_sim(
                environment=self.environment,
                reference_trajectory=self.reference_trajectory,
                model_type="LSTMForecaster",
                train_data=incomplete_train_data,
                hyperparams=self.hyperparams_lstm,
                control_params=self.control_params
            )

    def test_invalid_reference_trajectory_type(self):
        """Test invalid reference_trajectory type raises a TypeError."""
        with self.assertRaises(TypeError):
            closed_loop_sim(
                environment=self.environment,
                reference_trajectory=list(self.reference_trajectory),  # Should be a NumPy array
                model_type="LSTMForecaster",
                train_data=self.train_data,
                hyperparams=self.hyperparams_lstm,
                control_params=self.control_params
            )

if __name__ == "__main__":
    unittest.main()
