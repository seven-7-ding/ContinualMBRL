#!/bin/bash

# ============= Configuration =============
cd /home/jiale/MBRL/ContinualMBRL-primacy-bias

# Debug/deploy only on CUDA 6/7 for this primacy-bias sweep.
CUDA_DEVICES=(2 3 4 5)

TASK_STRING="finger_spin|walker_walk|cheetah_run|reacher_easy"

# Keep the wandb project identical to the full-loop baseline:
# logdir/<project>/<group>/seed_<seed>
PROJECT_PREFIX="continual_dreamer_full_loop"
GROUP_PREFIX="primacy_bias"

MODEL_SIZE="size1m"
BASE_LOGDIR_ROOT="logdir"

TRAIN_RATIO=1024
TASK_INTERVAL=200000
PRIM_COLLECT_STEPS=128
PRIM_TRAIN_STEPS=100000

declare -a SETTINGS=(
    # "only_wm|1000"
    # "only_wm|2000"
    # "only_wm|3000"

    # "only_agent_multi_rollout|1000"
    # "only_agent_multi_rollout|2000"
    # "only_agent_multi_rollout|3000"

    # "only_agent_one_rollout|1000"
    # "only_agent_one_rollout|2000"
    "only_agent_one_rollout|3000"

    # "both|1000"
    # "both|2000"
    # "both|3000"
)

run_counter=0
TOTAL_RUNS=${#SETTINGS[@]}
declare -a FAILED_DEPLOYMENTS=()

echo "============================================"
echo "Starting Continual DreamerV3 Priori Primacy-Bias Experiments"
echo "============================================"
echo "Total runs requested: $TOTAL_RUNS"
echo "Using GPUs: ${CUDA_DEVICES[@]}"
echo "Model size: $MODEL_SIZE"
echo "Tasks: $TASK_STRING"
echo "Prim collect steps: $PRIM_COLLECT_STEPS"
echo "Prim train steps: $PRIM_TRAIN_STEPS"
echo ""

for setting_spec in "${SETTINGS[@]}"; do
    IFS='|' read -r prim_mode seed <<< "$setting_spec"

    project="${PROJECT_PREFIX}_${MODEL_SIZE}"
    group="${GROUP_PREFIX}_${prim_mode}"
    logdir="$BASE_LOGDIR_ROOT/$project/$group/seed_$seed"
    if [ -s "$logdir/train.log" ]; then
        FAILED_DEPLOYMENTS+=("$setting_spec")
        echo "SKIPPED: $prim_mode with seed $seed (existing log: $logdir/train.log)"
        continue
    fi
    mkdir -p "$logdir"

    device_num="${CUDA_DEVICES[$run_counter % ${#CUDA_DEVICES[@]}]}"

    cmd_args=(
        python dreamerv3/main.py
        --configs continual_dmc_priori "$MODEL_SIZE"
        --task "$TASK_STRING"
        --logdir "$logdir"
        --run.train_ratio "$TRAIN_RATIO"
        --run.task_interval "$TASK_INTERVAL"
        --run.reset_on_switch False
        --run.reset_mode none
        --run.prim_mode "$prim_mode"
        --run.prim_collect_steps "$PRIM_COLLECT_STEPS"
        --run.prim_random_collect False
        --run.prim_train_steps "$PRIM_TRAIN_STEPS"
        --seed "$seed"
        --egl_device "$device_num"
        --agent.imag_length 15
        --agent.redo.redo_enabled True
        --agent.redo.grad_redo_enabled True
        --agent.redo.act_log_item log+erank+srank
        --agent.redo.grad_log_item log+erank+srank
    )

    echo "[$((run_counter + 1))/$TOTAL_RUNS] Launching: $prim_mode seed $seed -> GPU $device_num"
    echo "   Wandb project: $project"
    echo "   Wandb group: $group"
    echo "   Logdir: $logdir"
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$device_num "${cmd_args[@]}" > "$logdir/train.log" 2>&1 &

    run_counter=$((run_counter + 1))
    sleep 6
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
        IFS='|' read -r prim_mode seed <<< "$failed"
        echo "   - Prim Mode: $prim_mode, Seed: $seed"
    done
else
    echo "All experiments deployed successfully!"
fi

echo ""
echo "============================================"
echo "Monitoring Commands"
echo "============================================"
echo "  tail -f $BASE_LOGDIR_ROOT/${PROJECT_PREFIX}_${MODEL_SIZE}/${GROUP_PREFIX}_*/seed_*/train.log"
echo "  ps aux | grep 'dreamerv3/main.py'"
echo "  watch -n 1 nvidia-smi"
echo ""
