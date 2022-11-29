EPOCHS=50
LR=0.001
GAMMA=0.1
INIT_SIZE=0
AL_BSIZE=10
SAMPLE_METHOD=ae_coreset
DIM_REDUCTION=AE
HIDDEN_DIM1=64
HIDDEN_DIM2=32
DROOT=data/CIFAR10
DNAME=CIFAR10
OUT_DIR=output/
MAX_EPISODES=2
LOG_INTERVAL=10
CUDA_VISIBLE_DEVICES=0 python active_learn.py \
                      --epochs $EPOCHS --lr $LR \
                      --gamma $GAMMA --init-size $INIT_SIZE \
                      --al-batch-size $AL_BSIZE \
                      --sampling-method $SAMPLE_METHOD \
                      --dim-reduction $DIM_REDUCTION \
                      --hidden-dim1 $HIDDEN_DIM1 \
                      --hidden-dim2 $HIDDEN_DIM2 \
                      --dataset-name $DNAME \
                      --output-dir $OUT_DIR \
                      --max-eps $MAX_EPISODES \
                      --dataset-root $DROOT \
                      --log-interval $LOG_INTERVAL                  
