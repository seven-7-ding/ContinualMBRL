import jax.numpy as jnp

from jaxrl2.agents.sac_dreamer_dist_learner import FlaxWSC


def test_skip_last_layer_controls_rmsnorm_followed_dense_and_conv():
    params = {
        "encoder": {
            "Conv_0": {
                "kernel": jnp.ones((3, 3, 3, 8), dtype=jnp.float32),
                "bias": jnp.ones((8,), dtype=jnp.float32),
            },
            "Conv_0_norm": {
                "scale": jnp.ones((8,), dtype=jnp.float32),
            },
        },
        "Dense_0": {
            "kernel": jnp.ones((8, 16), dtype=jnp.float32),
            "bias": jnp.ones((16,), dtype=jnp.float32),
        },
        "Dense_0_norm": {
            "scale": jnp.ones((16,), dtype=jnp.float32),
        },
        "network": {
            "_SiLUMLP_0": {
                "layer_0": {
                    "kernel": jnp.ones((16, 32), dtype=jnp.float32),
                    "bias": jnp.ones((32,), dtype=jnp.float32),
                },
                "norm_0": {
                    "scale": jnp.ones((32,), dtype=jnp.float32),
                },
            },
            "Dense_0": {
                "kernel": jnp.ones((32, 4), dtype=jnp.float32),
                "bias": jnp.ones((4,), dtype=jnp.float32),
            },
        },
    }

    wsc = FlaxWSC(mechanism="wsc_skip_last_layer_dout_all", target="all")
    _new_params, metrics = wsc.apply(params, "actor")

    assert metrics["wsc/controlled/actor_encoder_Conv_0"] == 1.0
    assert metrics["wsc/controlled/actor_Dense_0"] == 1.0
    assert metrics["wsc/controlled/actor_network__SiLUMLP_0_layer_0"] == 1.0
    assert metrics["wsc/controlled/actor_network_Dense_0"] == 0.0


def test_skip_last_layer_constantinit_uses_initial_fnorm_over_8():
    params = {
        "Dense_0": {
            "kernel": jnp.ones((3, 4), dtype=jnp.float32) * 2.0,
            "bias": jnp.ones((4,), dtype=jnp.float32),
        },
        "Dense_0_norm": {
            "scale": jnp.ones((4,), dtype=jnp.float32),
        },
        "network": {
            "Dense_0": {
                "kernel": jnp.ones((4, 2), dtype=jnp.float32) * 3.0,
                "bias": jnp.ones((2,), dtype=jnp.float32),
            },
        },
    }

    wsc = FlaxWSC(
        mechanism="wsc_skip_last_layer_constantinit",
        target="all")
    new_params, metrics = wsc.apply(params, "actor")

    init_norm = jnp.sqrt(
        jnp.square(params["Dense_0"]["kernel"]).sum() +
        jnp.square(params["Dense_0"]["bias"]).sum())
    target = init_norm / 8.0
    new_norm = jnp.sqrt(
        jnp.square(new_params["Dense_0"]["kernel"]).sum() +
        jnp.square(new_params["Dense_0"]["bias"]).sum())

    assert jnp.allclose(metrics["wsc/target_norm/actor_Dense_0"], target)
    assert jnp.allclose(new_norm, target)
    assert metrics["wsc/controlled/actor_Dense_0"] == 1.0
    assert metrics["wsc/controlled/actor_network_Dense_0"] == 0.0
