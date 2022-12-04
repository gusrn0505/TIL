EPOCHS=100
LR=0.001
GAMMA=0.1
INIT_SIZE=0
AL_BSIZE=250
SAMPLE_METHOD=cae_coreset
DROOT=data/MNIST
DNAME=MNIST
OUT_DIR=output/
MAX_EPISODES=2
LOG_INTERVAL=10
CUDA_VISIBLE_DEVICES=0 python active_learn.py \
                      --epochs $EPOCHS --lr $LR \
                      --gamma $GAMMA --init-size $INIT_SIZE \
                      --al-batch-size $AL_BSIZE \
                      --sampling-method $SAMPLE_METHOD \
                      --dataset-name $DNAME \
                      --output-dir $OUT_DIR \
                      --max-eps $MAX_EPISODES \
                      --dataset-root $DROOT \
                      --log-interval $LOG_INTERVAL                  
