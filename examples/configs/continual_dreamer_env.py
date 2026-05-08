"""DreamerEnvLearner config.

Uses DreamerEnvLearner:
  - Actor : NormalTanhPolicySiLU (3-layer SiLU+RMSNorm)
  - Value : StateValueTwoHotSiLU (TwoHot distributional, 255 bins)
  - Training: Dreamer-style imagination via real env rollouts
  - No Q-function, no temperature; advantage-weighted PG + actent
"""
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # ---------- Network ----------
    config.hidden_dims  = (64, 64, 64)
    config.model_size   = 'size1m'
    config.num_bins     = 255

    # ---------- Learning rates ----------
    config.actor_lr = 3e-4
    config.value_lr = 3e-4

    # ---------- Dreamer training hyperparams ----------
    config.discount       = 0.997
    config.lam            = 0.95
    config.contdisc       = True
    config.horizon        = 333.0
    config.actent         = 3e-4
    config.slowreg        = 1.0
    config.slow_fraction  = 0.02
    config.imag_length    = 5

    config.jax_mem_fraction = 0.4

    # ---------- ReDo (mirrors dreamerv3/configs.yaml 'redo' block) ----------
    redo = ml_collections.ConfigDict()
    redo.redo_enabled      = True
    redo.grad_redo_enabled = False   # gradient ReDo not yet wired for DreamerEnv
    redo.tau               = 0.05
    redo.mode              = 'threshold'
    redo.frequency         = 10000
    redo.log_item          = 'log+erank+srank'
    redo.skip_last_layer   = False
    redo.rank_threshold    = 0.99
    redo.reset_start       = 0
    redo.reset_end         = 0
    config.redo = redo

    # ---------- Optimizer (mirrors dreamerv3/configs.yaml opt block) ----------
    opt = ml_collections.ConfigDict()
    opt.agc      = 0.3
    opt.eps      = 1e-20
    opt.beta1    = 0.9
    opt.beta2    = 0.999
    opt.momentum = True
    opt.nesterov = False
    opt.wd       = 0.0
    opt.wdregex  = '/kernel$'
    opt.schedule = 'const'
    opt.warmup   = 1000
    opt.anneal   = 0
    config.opt = opt

    return config
