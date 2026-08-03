# Mechanism Report

## Scope

This repo now uses `--run.reset_mechanism` as a target-parameter mechanism selector. The obsolete WSC layer-norm rescaling and per-layer `wsc_scale` output-scaling path were removed. The historical module name `embodied.jax.wsc` is kept for config/checkpoint compatibility, but the implementation is now `MechanismController`.

All requested experiment mechanisms target `all`.

## Shared Target Selection

Target selection is implemented in `embodied/jax/reset_targets.py` and used by `embodied/jax/wsc.py`.

For a parameter key `k`, the owning module path is `path(k)=k.rsplit("/", 1)[0]`. A parameter is selected if:

```text
not k.startswith(("opt/", "wsc/"))
and k is floating-point
and reset_targets.matches_target(path(k), target)
```

For this task, `target=all`, covering `enc`, `dyn`, `dec`, `rew`, `con`, `pol`, `val`, `slowval`, and normalizer modules.

## l2_decay

Formula:

```text
W_{t+1} = (1 - weight_decay) * W'_{t+1}
```

Here `W'_{t+1}` is the parameter after the normal optimizer update. This matches the AdamW-style decoupled decay requested by the user: it is not added as `||W||_2` to the loss.

Default hyperparameter:

```text
weight_decay = 2e-5
scale factor = 1 - 2e-5 = 0.99998
```

Pseudocode:

```text
grads = grad(task_loss)
params_after_opt = optimizer_apply(params, grads)
for each selected target parameter W:
    W = W * (1 - weight_decay)
```

Code mapping:

- `embodied/jax/wsc.py`: `MechanismController.step()`
- `embodied/jax/opt.py`: calls `step()` after `optax.apply_updates()`
- Metrics: `opt/mechanism/l2_decay/factor`, `opt/mechanism/l2_decay/weight_decay`, `opt/mechanism/target_param_l2`

## l2_init

Formula:

```text
L_init = lambda_init * 1/2 * sum_i ||W_i(t) - W_i(0)||_F^2
```

`W_i(0)` is stored at mechanism initialization for every selected target parameter.

Default hyperparameter:

```text
lambda_init = 2e-5
```

Pseudocode:

```text
if first creation:
    init_params = copy(selected target params)
loss = task_loss
for each selected target parameter W:
    loss += lambda_init * 0.5 * sum((W - init_params[W]) ** 2)
grads = grad(loss)
params = optimizer_apply(params, grads)
```

Code mapping:

- `embodied/jax/wsc.py`: `regularization_loss()`, `regularization_metrics()`
- `embodied/jax/opt.py`: adds the regularization inside `lossfn2()` before `nj.grad()`
- Metrics: `opt/mechanism/l2_init/raw_loss`, `opt/mechanism/l2_init/weighted_loss`
- Loss-namespace metrics: `loss/mechanism/l2_init/module_delta_sq/{module}` reports
  \(\sum_{i \in module} ||W_i(t)-W_i(0)||_F^2\) using the latest value in each log interval.

## continual_backprop

Reference behavior is based on the Continual Backprop repository `ellyhae/Continual-Backprop` and the CBP paper. The algorithm continually selects low-utility mature features and injects randomness by resetting their incoming weights while zeroing the corresponding outgoing fan-in to the next layer.

Sources:

- https://github.com/ellyhae/Continual-Backprop
- https://arxiv.org/abs/2108.06325

User-scoped implementation boundary:

```text
Only feature selection + weight reset are added.
Optimizer momentum, optimizer step-size computation, and model architecture are unchanged.
Optax optimizer state is not rewritten.
```

Default hyperparameters:

```text
eta = 0.99
maturity = 5000 optimizer updates
replacement_rate = 1e-4
eps = 1e-8
```

For each eligible layer pair `(current, next)`, let `h` be the current mean absolute activation per feature, `f` be its exponential trace, and `age` the feature age:

```text
age = age + 1
f_hat = f / (1 - eta ** age + eps)
f = eta * f + (1 - eta) * h

pre_w  = sum(abs(current_kernel), over incoming axes) + eps
post_w = sum(abs(next_kernel), over outgoing axes from current feature)
y = abs(h - f_hat) * post_w / pre_w
u = eta * u + (1 - eta) * y
u_hat = u / (1 - eta ** age + eps)
```

Feature replacement:

```text
eligible = age > maturity
with probability min(1, num_features * replacement_rate):
    r = argmin(u_hat over eligible features)
    reset current_kernel[..., r] with LeCun/truncated-normal samples
    reset current_bias[r] to 0 if present
    set next_kernel[r, ...] or next_kernel[..., r, :] to 0
    set age[r], f[r], u[r] to 0
```

Code mapping:

- `dreamerv3/agent.py`: when mechanism is `continual_backprop`, runs a no-gradient activation pass after normal training and calls `continual_backprop_step()`
- `dreamerv3/rssm.py` and `embodied/jax/nets.py`: existing `LAYER_CALLBACK` sites provide layer activations
- `embodied/jax/wsc.py`: `continual_backprop_step()`
- Metrics: `train/mechanism/cbp/eligible_layers`, `train/mechanism/cbp/replaced/*`, `train/mechanism/cbp/min_utility/*`, `train/mechanism/cbp/mean_age/*`
- Interval reset-count metrics: `train/mechanism/cbp/reset_count_since_log/{layer}`
  is summed by the training logger, so each log step reports how many neurons
  were reset in that layer since the previous log. Repeated resets of the same
  neuron in the interval are counted repeatedly.

## Config And CLI

Default config keys in `dreamerv3/configs.yaml`:

```yaml
run.reset_mechanism: disabled
run.reset_target: all
agent.wsc.weight_decay: 2e-5
agent.wsc.l2_init_weight: 2e-5
agent.wsc.cbp_eta: 0.99
agent.wsc.cbp_maturity: 5000
agent.wsc.cbp_replacement_rate: 1e-4
agent.wsc.cbp_eps: 1e-8
agent.redo.redo_enabled: false
agent.redo.grad_redo_enabled: false
```

ReDo metric reporting is enabled for the formal mechanism experiments with
`agent.redo.redo_enabled=True`, `agent.redo.grad_redo_enabled=True`,
`agent.redo.act_log_item=log+erank+srank`, and
`agent.redo.grad_log_item=log+erank+srank`. These settings compute and upload
`act_redo/*`, `grad_redo/*`, and `data_diversity/*` metrics without including
the `reset` action in either ReDo analyser.

## Validation

Commands run in the existing conda env `dreamer`:

```bash
conda run -n dreamer python -m py_compile embodied/jax/wsc.py embodied/jax/__init__.py dreamerv3/agent.py embodied/tests/test_wsc.py embodied/jax/opt.py embodied/jax/nets.py embodied/run/continual_train.py embodied/jax/reset_targets.py
conda run -n dreamer python -c "from embodied.tests import test_wsc; names=sorted(n for n in dir(test_wsc) if n.startswith('test_')); [print('RUN', n) or getattr(test_wsc, n)() for n in names]; print('OK')"
```

Direct tests covered:

- `l2_decay` scales selected `all` target params and leaves non-target params unchanged.
- `l2_init` stores initialization params and computes the expected weighted Frobenius penalty.
- `continual_backprop` selects a mature low-utility unit and resets weights.

Smoke training used `dummy_disc` debug runs with `batch_size=2`, `batch_length=4`, `replay_context=0`, and `run.steps=80`. All three mechanisms completed actual optimizer updates:

- `l2_decay`: final metrics included `opt/mechanism/l2_decay/*`, `opt/updates ~= 59.5`.
- `l2_init`: final metrics included `opt/mechanism/l2_init/raw_loss` and `weighted_loss`, `opt/updates ~= 59.5`.
- `continual_backprop`: final metrics included `train/mechanism/cbp/*`, `opt/updates ~= 59.5`.
