# AE, CAE 주어진 데이터 셋에 대해서 학습하기 
EPOCHS=100
LR=0.001
GAMMA=0.1
INIT_SIZE=0
AL_BSIZE=100
SAMPLE_METHOD=cae_coreset
DROOT=data/MNIST
DNAME=MNIST
OUT_DIR=output/
MAX_EPISODES=2
CUDA_VISIBLE_DEVICES=0 python learn_autoencoder.py \
                      --epochs $EPOCHS --lr $LR \
                      --gamma $GAMMA --init-size $INIT_SIZE \
                      --al-batch-size $AL_BSIZE \
                      --sampling-method $SAMPLE_METHOD \
                      --dataset-name $DNAME \
                      --output-dir $OUT_DIR \
                      --max-eps $MAX_EPISODES \
                      --dataset-root $DROOT  
