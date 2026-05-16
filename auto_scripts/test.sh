cd /home/jiale/MBRL/ContinuousMBRL
TASK=dmc_walker_walk
SEED=1000
LOGDIR=logdir/dreamer_${TASK}/debug/seed_${SEED}
DEVICE=7
mkdir -p ${LOGDIR}

CUDA_VISIBLE_DEVICES=${DEVICE} python dreamerv3/main.py \
    --configs dmc_vision size12m \
    --task ${TASK} \
    --logdir ${LOGDIR} \
    --run.train_ratio 32 \
    --seed ${SEED} \
    --egl_device ${DEVICE} \
    > ${LOGDIR}/train.log 2>&1 &