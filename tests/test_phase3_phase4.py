from __future__ import annotations

import numpy as np
import torch

from predictors.gnn_predictor import GNNModelConfig, GNNPredictor, build_distance_adjacency, build_graph_windows
from predictors.lstm_predictor import LSTMModelConfig, LSTMPredictor, build_sequence_windows


def test_lstm_predictor_forward_shape():
    trace = np.random.default_rng(0).random((20, 6), dtype=np.float32)
    x, y = build_sequence_windows(trace, 5, 1, target_size=3)
    model = LSTMPredictor(LSTMModelConfig(input_size=6, output_size=3, hidden_size=8, num_layers=1, dropout=0.0))
    output = model(torch.as_tensor(x[:2], dtype=torch.float32))
    assert output.shape == (2, 3)
    assert y.shape[1] == 3


def test_gnn_predictor_forward_shape():
    trace = np.random.default_rng(1).random((20, 4, 5), dtype=np.float32)
    x, y = build_graph_windows(trace, 5, 1, target_columns=(0, 1))
    adjacency = torch.as_tensor(build_distance_adjacency(np.random.default_rng(2).random((4, 2)), 2), dtype=torch.float32)
    model = GNNPredictor(GNNModelConfig(input_size=25, output_size=2, hidden_size=8, num_layers=1, dropout=0.0))
    output = model(torch.as_tensor(x[:2], dtype=torch.float32), adjacency)
    assert output.shape == (2, 4, 2)
    assert y.shape[1:] == (4, 2)


def test_mec_env_reset_and_step():
    import pytest
    pytest.importorskip("gymnasium")
    from rl.mec_env import MECEnvConfig, MECOffloadingEnv
    env = MECOffloadingEnv("configs/phase1_small.yaml", MECEnvConfig(top_k=3, max_episode_tasks=5))
    observation, info = env.reset(seed=0)
    assert observation.shape == env.observation_space.shape
    action = env.action_space.sample()
    next_observation, reward, terminated, truncated, step_info = env.step(action)
    assert next_observation.shape == env.observation_space.shape
    assert isinstance(float(reward), float)
    assert not truncated
    assert "success" in step_info
    env.close()
