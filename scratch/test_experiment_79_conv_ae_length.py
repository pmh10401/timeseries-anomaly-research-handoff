import torch

from run_experiment_79_vae_epoch_sweep_guarded import ConvAutoEncoderFactory


def test_conv_autoencoder_preserves_problem_lengths():
    model = ConvAutoEncoderFactory(torch.nn).build()
    for length in [46, 270, 301, 315]:
        x = torch.randn(2, 1, length)
        y = model(x)
        assert y.shape == x.shape
