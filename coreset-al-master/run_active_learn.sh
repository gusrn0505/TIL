EPOCHS=50
LR=0.001
GAMMA=0.1
INIT_SIZE=10
AL_BSIZE=100
SAMPLE_METHOD=prob_uncertain
DROOT=data/mnist_easy
DNAME=mnist
OUT_DIR=output/
MAX_EPISODES=10
CUDA_VISIBLE_DEVICES=0 python active_learn.py \
                      --epochs $EPOCHS --lr $LR \
                      --gamma $GAMMA --init-size $INIT_SIZE \
                      --al-batch-size $AL_BSIZE \
                      --sampling-method $SAMPLE_METHOD \
                      --dataset-name $DNAME \
                      --output-dir $OUT_DIR \
                      --max-eps $MAX_EPISODES \
                      --dataset-root $DROOT  
