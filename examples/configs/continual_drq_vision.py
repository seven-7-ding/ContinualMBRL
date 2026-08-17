"""Continual DrQ vision config.

Uses DrQLearner with Dreamer-aligned size presets for actor/critic MLP heads,
Adam optimizer, optional WSC, optional L2-init regularization, and an explicit
off-by-default DrQ-v2 random-shift mechanism for future augmentation runs.
"""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 4e-5
    config.critic_lr = 4e-5
    config.temp_lr = 4e-5

    config.hidden_dims = (64, 64, 64)
    config.model_size = "size1m"

    config.cnn_features = (32, 64, 128, 256)
    config.cnn_filters = (3, 3, 3, 3)
    config.cnn_strides = (2, 2, 2, 2)
    config.cnn_padding = "VALID"
    config.latent_dim = 50
    config.encoder = "d4pg"

    config.discount = 0.99
    config.tau = 0.005
    config.init_temperature = 0.1
    config.target_entropy = None
    config.backup_entropy = True
    config.critic_reduction = "mean"
    config.augmentation_enabled = False
    config.augmentation_pad = 4
    config.jax_mem_fraction = 0.25

    opt = ml_collections.ConfigDict()
    opt.optimizer = "adam"
    opt.agc = 0.3
    opt.eps = 1e-20
    opt.beta1 = 0.9
    opt.beta2 = 0.999
    opt.momentum = True
    opt.nesterov = False
    opt.wd = 0.0
    opt.wdregex = "/kernel$"
    opt.schedule = "const"
    opt.warmup = 1000
    opt.anneal = 0
    config.opt = opt

    wsc = ml_collections.ConfigDict()
    wsc.mechanism = "disabled"
    wsc.target = "all"
    wsc.eps = 1e-8
    wsc.factor_min = 0.01
    wsc.factor_max = 100.0
    config.wsc = wsc

    l2_init = ml_collections.ConfigDict()
    l2_init.enabled = False
    l2_init.weight = 2e-5
    config.l2_init = l2_init

    redo = ml_collections.ConfigDict()
    redo.grad_redo_enabled = True
    redo.grad_redo_frequency = 1000
    redo.grad_redo_reset_start = 0
    redo.grad_redo_reset_end = 0
    redo.grad_redo_skip_last_layer = False
    config.redo = redo

    return config
