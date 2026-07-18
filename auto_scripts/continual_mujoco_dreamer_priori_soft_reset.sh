#!/bin/bash

# ============= Configuration =============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Available CUDA devices for this experiment.
CUDA_DEVICES=(0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7)

# Maximum concurrent runs launched by this script on each GPU.
MAX_RUNS_PER_GPU=1

# Task string (same for all settings)
TASK_STRING="walker_run|hopper_hop|fish_swim"

# Prefix for log directories
PREFIX="continual_dreamer_soft_reset"

# Model configuration
MODEL_SIZE="size1m"  # Options: size0.5m, size1m, size12m, size50m, etc.

# Base log directory
BASE_LOGDIR_ROOT="logdir"

# Training configuration
TRAIN_RATIO=1024
TASK_INTERVAL=1000000
RESET_FREQUENCY=0
RESET_TARGET="all"
WSC_TARGET_NORM=1.0
WSC_SCALE_FACTOR=0.999

REVIVE_EPOCH=0
REVIVE_STRATEGY="threshold"

# Other hyper-parameters
AGENT_IMAG_LENGTH=15
REDO_ENABLED=True
GRAD_REDO_ENABLED=True
ACT_LOG_ITEM="log+erank+srank"
GRAD_LOG_ITEM="log+erank+srank"

# Optional extra CLI args, for example:
# EXTRA_ARGS="--run.log_every 5000 --batch_size 8"
EXTRA_ARGS=""

# ============= Settings Definition =============
# Format: "wsc_mechanism|seed"
declare -a SETTINGS=(
    "no_wsc|1000"
    "no_wsc|2000"
    "no_wsc|3000"
    "WSC_nograd_scale_init|1000"
    "WSC_nograd_scale_init|2000"
    "WSC_nograd_scale_init|3000"
    "WSC_grad_scale_init|1000"
    "WSC_grad_scale_init|2000"
    "WSC_grad_scale_init|3000"
    "WSC_nograd_scale_constant|1000"
    "WSC_nograd_scale_constant|2000"
    "WSC_nograd_scale_constant|3000"
    "WSC_grad_scale_constant|1000"
    "WSC_grad_scale_constant|2000"
    "WSC_grad_scale_constant|3000"
    "WSC_nograd_scale_factor|1000"
    "WSC_nograd_scale_factor|2000"
    "WSC_nograd_scale_factor|3000"
    "WSC_grad_scale_factor|1000"
    "WSC_grad_scale_factor|2000"
    "WSC_grad_scale_factor|3000"
)

# ============= Initialize =============
run_counter=0
TOTAL_RUNS=${#SETTINGS[@]}
declare -a FAILED_DEPLOYMENTS=()

# ============= Run Experiments =============
echo "============================================"
echo "Starting Continual DreamerV3 Priori WSC Experiments"
echo "============================================"
echo "Total runs requested: $TOTAL_RUNS"
echo "Using GPUs: ${CUDA_DEVICES[@]}"
echo "Max runs per GPU launched by this script: $MAX_RUNS_PER_GPU"
echo "Model size: $MODEL_SIZE"
echo "Tasks: $TASK_STRING"
echo "WSC target: $RESET_TARGET"
echo "WSC target norm: $WSC_TARGET_NORM"
echo "WSC scale factor: $WSC_SCALE_FACTOR"
if (( REVIVE_EPOCH > 0 )); then
    echo "Revive schedule: up to $REVIVE_EPOCH updates after each reset ($REVIVE_STRATEGY)"
else
    echo "Revive schedule: disabled (REVIVE_EPOCH=0)"
fi
echo ""

# Iterate over all settings
for setting_spec in "${SETTINGS[@]}"; do
    IFS='|' read -r wsc_mechanism seed <<< "$setting_spec"
    reset_tag="wsc_${wsc_mechanism}_${RESET_TARGET}"
    reset_mechanism="$wsc_mechanism"
    reset_target="$RESET_TARGET"
    if [[ "$wsc_mechanism" == "no_wsc" || "$wsc_mechanism" == "disabled" || "$wsc_mechanism" == "off" || "$wsc_mechanism" == "none" ]]; then
        reset_tag="no_wsc"
        reset_mechanism="disabled"
        reset_target="all"
    fi

    # Create log directory
    logdir="$BASE_LOGDIR_ROOT/${PREFIX}_${MODEL_SIZE}/${reset_tag}/seed_$seed"
    if [[ "${FRESH_RERUN:-0}" == "1" && -e "$logdir" ]]; then
        backup="${logdir}.failed.$(date +%Y%m%d_%H%M%S)"
        echo "FRESH_RERUN moving existing $logdir -> $backup"
        mv "$logdir" "$backup"
    fi
    if [ -s "$logdir/train.log" ]; then
        FAILED_DEPLOYMENTS+=("$setting_spec")
        echo "SKIPPED: $wsc_mechanism with seed $seed (existing log: $logdir/train.log)"
        continue
    fi
    mkdir -p "$logdir"

    # Assign GPU by sequential index (no waiting).
    device_num="${CUDA_DEVICES[$run_counter % ${#CUDA_DEVICES[@]}]}"

    # Use array to avoid leading-space issues from multi-line string quoting.
    cmd_args=(
        python dreamerv3/main.py
        --configs continual_dmc_priori "$MODEL_SIZE"
        --task "$TASK_STRING"
        --logdir "$logdir"
        --run.train_ratio "$TRAIN_RATIO"
        --run.task_interval "$TASK_INTERVAL"
        --run.reset_frequency "$RESET_FREQUENCY"
        --run.reset_mechanism "$reset_mechanism"
        --run.reset_target "$reset_target"
        --run.revive_epoch "$REVIVE_EPOCH"
        --run.revive_strategy "$REVIVE_STRATEGY"
        --agent.wsc.target_norm "$WSC_TARGET_NORM"
        --agent.wsc.scale_factor "$WSC_SCALE_FACTOR"
        --seed "$seed"
        --egl_device "$device_num"
        --agent.imag_length "$AGENT_IMAG_LENGTH"
        --agent.redo.redo_enabled "$REDO_ENABLED"
        --agent.redo.grad_redo_enabled "$GRAD_REDO_ENABLED"
        --agent.redo.act_log_item "$ACT_LOG_ITEM"
        --agent.redo.grad_log_item "$GRAD_LOG_ITEM"
    )

    if [[ -n "$EXTRA_ARGS" ]]; then
        read -r -a extra_args_array <<< "$EXTRA_ARGS"
        cmd_args+=("${extra_args_array[@]}")
    fi

    # Execute
    echo "[$((run_counter + 1))/$TOTAL_RUNS] Launching: $wsc_mechanism target $RESET_TARGET seed $seed -> GPU $device_num"
    echo "   Task order: $TASK_STRING"
    echo "   Config: continual_dmc_priori $MODEL_SIZE reset_mechanism=$reset_mechanism reset_target=$reset_target reset_frequency=$RESET_FREQUENCY revive_epoch=$REVIVE_EPOCH"
    echo "   ReDo: act_log_item=$ACT_LOG_ITEM grad_log_item=$GRAD_LOG_ITEM"
    if [[ -n "$EXTRA_ARGS" ]]; then
        echo "   Extra args: $EXTRA_ARGS"
    fi
    echo "   Logdir: $logdir"
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$device_num "${cmd_args[@]}" > "$logdir/train.log" 2>&1 &

    run_counter=$((run_counter + 1))
    sleep 50
    echo ""
done

echo ""
echo "============================================"
echo "Deployment Summary"
echo "============================================"
echo "Successfully deployed: $run_counter / $TOTAL_RUNS"

if [ ${#FAILED_DEPLOYMENTS[@]} -gt 0 ]; then
    echo "Failed to deploy: ${#FAILED_DEPLOYMENTS[@]} experiments"
    for failed in "${FAILED_DEPLOYMENTS[@]}"; do
        IFS='|' read -r wsc_mechanism seed <<< "$failed"
        echo "   - WSC Mechanism: $wsc_mechanism, Seed: $seed"
    done
else
    echo "All experiments deployed successfully!"
fi

echo ""
echo "============================================"
echo "Monitoring Commands"
echo "============================================"
echo "  tail -f $BASE_LOGDIR_ROOT/${PREFIX}_${MODEL_SIZE}/*/seed_*/train.log"
echo "  ps aux | grep 'dreamerv3/main.py'"
echo "  watch -n 1 nvidia-smi"
echo ""
