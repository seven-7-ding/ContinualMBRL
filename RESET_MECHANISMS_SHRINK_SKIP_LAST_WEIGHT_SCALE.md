# `shrink_skip_last` and `weight_scale` reset report

## Scope and conclusions

This report describes the two reset mechanisms implemented in
`embodied/jax/agent.py` for the `size1m` priori model used by
`walker_run|hopper_hop|fish_swim`.

The previous `shrink` implementation did **not** satisfy the new requirement.
It multiplied every selected floating-point parameter by `RESET_ALPHA`,
including predictive output projections and RMSNorm scales. The old name
`shrink` is now a compatibility alias for `shrink_skip_last`; new runs use the
new name explicitly.

Both new mechanisms now preserve:

- predictive output projection parameters;
- RMSNorm `scale` parameters;
- optimizer states corresponding to those preserved parameters;
- the optimizer global update counter and learning-rate scheduler state.

## Why RMSNorm is not scaled

The MLP order in `embodied/jax/nets.py` is:

```text
y = W x + b
z = RMSNorm(y; gamma)
a = activation(z)
```

Ignoring the small RMSNorm epsilon, scaling `W` and `b` by a positive factor
`c` gives:

```text
RMSNorm(c y; gamma) ~= RMSNorm(y; gamma)
```

Thus, keeping `gamma` unchanged approximately preserves the scale entering the
activation. If `gamma` were also scaled by `c`, then:

```text
RMSNorm(c y; c gamma) ~= c * RMSNorm(y; gamma)
```

That would *not* preserve the activation input scale. Therefore both new
mechanisms exclude parameters such as `val/mlp/norm0/scale` and
`dyn/dynin0norm/scale`.

## Shared reset flow

1. `embodied/run/continual_train.py::periodic_reset()` triggers at each
   `reset_frequency` boundary and calls `agent.reset_params()` with the chosen
   target, mechanism, and alpha.
2. `embodied/jax/agent.py::reset_params()` canonicalizes the mechanism. Legacy
   `shrink` and `shrink_only` names map to `shrink_skip_last`.
3. A fresh parameter/optimizer tree is initialized by `_init_params()`. For
   these two mechanisms, fresh model weights are not copied into the model;
   the fresh tree supplies only matching freshly initialized optimizer state.
4. `_reset_key_matches()` applies the selected reset target. For the production
   experiments the target is `all`, so all modules are initially candidates.
5. `_is_output_layer_param()` removes output projections whose final module
   name is one of `logit`, `logits`, `obslogit`, `priorlogit`, `mean`,
   `stddev`, `pred`, or `imgout`.
6. `_is_norm_param()` removes RMSNorm `scale`/`shift` parameters.
7. The mechanism updates the remaining parameter values.
8. `_opt_state_matches_param_reset()` selects optimizer arrays ending in each
   changed model parameter key. Only those per-parameter optimizer arrays are
   replaced by fresh values. The optimizer global counter is not reset.
9. Policy-device copies are refreshed from the updated training parameters.
10. `revive_epoch=0` skips revive, matching the requested experiment setup.

## `shrink_skip_last`

For every selected floating-point parameter that is neither an output
projection nor an RMSNorm parameter:

```text
theta' = RESET_ALPHA * theta
```

For the production runs, `RESET_ALPHA=0.8`. Non-floating counters are copied
unchanged. The two floating return-normalizer bounds are selected by target
`all` and therefore are also multiplied by 0.8:

- `retnorm/hi/value`
- `retnorm/lo/value`

The mechanism changes 52 floating arrays containing 636,738 scalar values.
Of these, 50 arrays (636,736 values) are kernels and biases, and 2 values are
the return-normalizer bounds.

## `weight_scale`

At initial agent construction, every non-output layer's kernel and bias are
grouped by layer path. For a layer with kernel `W` and bias `b`, the stored
initial Frobenius norm is:

```text
f_0 = sqrt(sum(W_0 ** 2) + sum(b_0 ** 2))
```

This is equivalent to the Frobenius norm of a conceptual concatenation of `W`
and `b`; the concatenation orientation does not affect the norm. The complete
`initial_layer_norms` dictionary is stored in each agent checkpoint, so resume
uses the original run's norms rather than recomputing them from trained
weights.

At reset step `t`, for each selected non-output layer:

```text
f_t = sqrt(sum(W_t ** 2) + sum(b_t ** 2))
scale_t = f_0 / f_t
W' = scale_t * W_t
b' = scale_t * b_t
```

Kernel and bias always use the same scalar factor. If `f_t` is exactly zero,
the implementation uses factor 1 to avoid division by zero; normal trained
layers do not reach this branch. RMSNorm parameters, output projections, and
non-layer state such as return-normalizer bounds are unchanged.

`RESET_ALPHA` is accepted and logged for a uniform experiment interface but
does not enter the `weight_scale` formula.

The mechanism changes 50 arrays containing 636,736 scalar values.

## Affected kernel/bias layers for `reset_target=all`

Each listed layer includes both `kernel` and `bias`.

| Module | Layer paths | Scalar values |
| --- | --- | ---: |
| encoder | `enc/mlp0`, `enc/mlp1`, `enc/mlp2` | 10,432 |
| dynamics | `dyn/dyngru`, `dyn/dynhid0`, `dyn/dynin0`, `dyn/dynin1`, `dyn/dynin2`, `dyn/obs0`, `dyn/prior0`, `dyn/prior1` | 346,880 |
| decoder | `dec/mlp/linear0`, `dec/mlp/linear1`, `dec/mlp/linear2` | 49,344 |
| reward head | `rew/mlp/linear0` | 41,024 |
| continuation head | `con/mlp/linear0` | 41,024 |
| policy head | `pol/mlp/linear0`, `pol/mlp/linear1`, `pol/mlp/linear2` | 49,344 |
| value head | `val/mlp/linear0`, `val/mlp/linear1`, `val/mlp/linear2` | 49,344 |
| target value head | `slowval/mlp/linear0`, `slowval/mlp/linear1`, `slowval/mlp/linear2` | 49,344 |
| **Total** | 25 layers / 50 arrays | **636,736** |

## Explicitly skipped output parameters

The following 18 arrays, totaling 69,290 values, are unchanged by both
mechanisms and keep their optimizer state:

```text
con/head/logit/{kernel,bias}
dec/vec/state/pred/{kernel,bias}
dyn/obslogit/{kernel,bias}
dyn/priorlogit/{kernel,bias}
pol/head/action/mean/{kernel,bias}
pol/head/action/stddev/{kernel,bias}
rew/head/logits/{kernel,bias}
slowval/head/logits/{kernel,bias}
val/head/logits/{kernel,bias}
```

This includes the policy mean/stddev projections requested in the prompt and
the value/reward two-hot logits projections used by the current configuration.

## Explicitly skipped RMSNorm parameters

The following 24 RMSNorm scale arrays, totaling 1,984 values, are unchanged by
both mechanisms and keep their optimizer state:

```text
enc/{mlp0norm,mlp1norm,mlp2norm}/scale
dyn/{dynhid0norm,dynin0norm,dynin1norm,dynin2norm,obs0norm,
     prior0norm,prior1norm}/scale
dec/mlp/{norm0,norm1,norm2}/scale
rew/mlp/norm0/scale
con/mlp/norm0/scale
pol/mlp/{norm0,norm1,norm2}/scale
val/mlp/{norm0,norm1,norm2}/scale
slowval/mlp/{norm0,norm1,norm2}/scale
```

## Optimizer behavior

For the current size1m parameter tree, both mechanisms match 44 trainable
kernel/bias arrays to optimizer state. The target-value (`slowval`) copy is
updated as a model parameter but has no direct optimizer state. The matching
optimizer reset replaces 88 state arrays (1,174,784 scalar values),
corresponding to the optimizer's two per-parameter state tensors for those 44
trainable arrays.

The following are preserved:

- output-layer optimizer state;
- RMSNorm optimizer state;
- optimizer state for every parameter outside the chosen reset target;
- optimizer global step/counter and scheduler progression.

## Validation performed

- Python compilation passed for `embodied/jax/agent.py` and
  `embodied/run/continual_train.py`.
- Helper tests verified output-layer and RMSNorm exclusion.
- A numerical test changed a layer's joint norm from 10 to its stored initial
  value 5 and verified the resulting norm exactly.
- A size0.5m CUDA run executed `weight_scale` at agent step 1008 and continued
  without host-to-device transfer or deleted-array errors.
- A size0.5m CUDA run executed `shrink_skip_last` at agent step 1008 and
  continued without reset errors.
- Checkpoints produced by both tests contained `initial_layer_norms` with 25
  layer entries.
- All three production `shrink_skip_last` seeds completed the first reset at
  step 50,000 and subsequently reported at least step 70,000.
- All three production `weight_scale` seeds completed the first reset at step
  50,000. Two runs initially exposed a `ptxas` crash while JAX compiled many
  one-off square/sum reductions concurrently. The norm reduction now uses an
  explicit device-to-host copy and NumPy float64 accumulation at sparse reset
  points, followed by one scalar device placement per layer. This preserves the
  normalization formula while avoiding those temporary GPU kernels. Both
  affected runs completed step 50,000 reset after resuming their step 40,000
  checkpoints with the fix. All three production seeds subsequently reported
  step 70,000 while running the fixed implementation.
- All six production runs completed their second reset at step 100,000. The
  latest log for every seed contains the corresponding
  `Reset full system at step 100000` marker without a fatal error. A direct
  W&B API check showed all six remote runs in `running` state with the correct
  `performance/0_walker_run/score` key; remote steps were within one 10,000-step
  reporting window of local progress.
- All six production runs also completed their third reset at step 150,000;
  every seed contains the expected mechanism-specific completion marker and
  continued training afterward, with the leading runs reporting step 160,000.
- All six production runs completed their eighteenth reset at step 900,000 with
  the expected mechanism-specific completion markers and no fatal process.
  Every run continued to report higher steps after this reset. At the time of
  this check, the slowest production run had reached step 910,000 and the
  fastest had reached step 1,040,000.
- All six production runs completed the reset at step 1,000,000 and switched
  from `walker_run` to `hopper_hop` with the expected padded observation/action
  dimensions. Each run logged the mechanism-specific reset marker, skipped
  revive because `revive_epoch=0`, emitted `Task: hopper_hop`, and saved a
  checkpoint after the environment switch. All six runs have already reported
  `performance/1_hopper_hop/score`.

## Production experiment configuration

Both mechanisms use three seeds (`1000`, `2000`, `3000`) with:

```text
task: walker_run|hopper_hop|fish_swim
model: size1m
steps: 500000000
train_ratio: 1024
task_interval: 1000000
reset_target: all
reset_frequency: 50000
reset_alpha: 0.8
revive_epoch: 0
replay.chunksize: 1024
replay.cache_chunks: 4096
```

Logs and replay chunks remain inside each run directory under:

```text
logdir/continual_dreamer_soft_reset_size1m/
  shrink_skip_last_all_a0p8_50k_no_revive/seed_*/
  weight_scale_all_a0p8_50k_no_revive/seed_*/
```

Initial production deployment:

| Mechanism | Seed | GPU | W&B run ID |
| --- | ---: | ---: | --- |
| `shrink_skip_last` | 1000 | 0 | `zgap7ob0` |
| `shrink_skip_last` | 2000 | 5 | `w0jnerd4` |
| `shrink_skip_last` | 3000 | 6 | `xi2x6kv3` |
| `weight_scale` | 1000 | 2 | `4k3nyr3f` |
| `weight_scale` | 2000 | 1 | `3153pyty` |
| `weight_scale` | 3000 | 7 | `4od2yrfh` |
