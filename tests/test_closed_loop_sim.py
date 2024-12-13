import unittest
import torch
from src.RNNmpc.utils import closed_loop_sim


class TestClosedLoopSim(unittest.TestCase):

    def setUp(self):
        # Create a mock simulator and forecaster for testing
        class MockSimulator:
            def __init__(self):
                self.control_dim = 1
                self.sensor_dim = 2

            def initial_state(self):
                return torch.zeros((2, 1), dtype=torch.float64)

            def simulate(self, U, x0):
                return x0 + U

        class MockForecaster:
            def forecast(self, U, **kwargs):
                return U * 2

        self.simulator = MockSimulator()
        self.forecaster = MockForecaster()

        self.controller_params = {
            "dev": 100,
            "u_1": 1,
            "u_2": 10,
            "soft_bounds": (0.05, 0.95),
            "hard_bounds": (0, 1),
        }

        self.ref_traj = torch.ones((2, 100), dtype=torch.float64)
        self.forecast_horizon = 10
        self.control_horizon = 5
        self.num_steps = 20

    def test_valid_inputs(self):
        # Ensure function runs without error on valid inputs
        results = closed_loop_sim(
            simulator=self.simulator,
            forecaster=self.forecaster,
            controller_params=self.controller_params,
            ref_traj=self.ref_traj,
            forecast_horizon=self.forecast_horizon,
            control_horizon=self.control_horizon,
            num_steps=self.num_steps,
        )

        self.assertIn("U", results)
        self.assertIn("S", results)
        self.assertIn("ref_traj", results)
        self.assertEqual(results["U"].shape[1], self.num_steps)
        self.assertEqual(results["S"].shape[1], self.num_steps)

    def test_invalid_simulator(self):
        with self.assertRaises(TypeError):
            closed_loop_sim(
                simulator=None,
                forecaster=self.forecaster,
                controller_params=self.controller_params,
                ref_traj=self.ref_traj,
                forecast_horizon=self.forecast_horizon,
                control_horizon=self.control_horizon,
                num_steps=self.num_steps,
            )

    def test_invalid_forecaster(self):
        with self.assertRaises(TypeError):
            closed_loop_sim(
                simulator=self.simulator,
                forecaster=None,
                controller_params=self.controller_params,
                ref_traj=self.ref_traj,
                forecast_horizon=self.forecast_horizon,
                control_horizon=self.control_horizon,
                num_steps=self.num_steps,
            )

    def test_invalid_ref_traj(self):
        with self.assertRaises(TypeError):
            closed_loop_sim(
                simulator=self.simulator,
                forecaster=self.forecaster,
                controller_params=self.controller_params,
                ref_traj=None,
                forecast_horizon=self.forecast_horizon,
                control_horizon=self.control_horizon,
                num_steps=self.num_steps,
            )

    def test_negative_forecast_horizon(self):
        with self.assertRaises(ValueError):
            closed_loop_sim(
                simulator=self.simulator,
                forecaster=self.forecaster,
                controller_params=self.controller_params,
                ref_traj=self.ref_traj,
                forecast_horizon=-1,
                control_horizon=self.control_horizon,
                num_steps=self.num_steps,
            )

    def test_negative_control_horizon(self):
        with self.assertRaises(ValueError):
            closed_loop_sim(
                simulator=self.simulator,
                forecaster=self.forecaster,
                controller_params=self.controller_params,
                ref_traj=self.ref_traj,
                forecast_horizon=self.forecast_horizon,
                control_horizon=-1,
                num_steps=self.num_steps,
            )

    def test_negative_num_steps(self):
        with self.assertRaises(ValueError):
            closed_loop_sim(
                simulator=self.simulator,
                forecaster=self.forecaster,
                controller_params=self.controller_params,
                ref_traj=self.ref_traj,
                forecast_horizon=self.forecast_horizon,
                control_horizon=self.control_horizon,
                num_steps=-1,
            )

    def test_large_forecast_horizon(self):
        # Ensure function runs when forecast_horizon is very large
        large_forecast_horizon = 50
        results = closed_loop_sim(
            simulator=self.simulator,
            forecaster=self.forecaster,
            controller_params=self.controller_params,
            ref_traj=self.ref_traj,
            forecast_horizon=large_forecast_horizon,
            control_horizon=self.control_horizon,
            num_steps=self.num_steps,
        )

        self.assertIn("U", results)
        self.assertIn("S", results)
        self.assertIn("ref_traj", results)
        self.assertEqual(results["U"].shape[1], self.num_steps)
        self.assertEqual(results["S"].shape[1], self.num_steps)


if __name__ == "__main__":
    unittest.main()
