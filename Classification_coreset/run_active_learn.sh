F_EPOCHS=10
S_EPOCHS=5
T_EPOCHS=5
N_CLASS=10
LR=0.001
GAMMA=0.5

SAMPLING_SIZE=1000
MAX_EPISODES=1

DNAME=MNIST
OUT_DIR=output/

NUM_SUBGRAPH=0
MIN_DENSITY=0
MAX_DENSITY=0
THRESHOLD=0.5


CUDA_VISIBLE_DEVICES=0 python active_learn.py \
                      --first-epochs $F_EPOCHS \
                      --second-epochs $S_EPOCHS \
                      --third-epochs $T_EPOCHS \
                      --num-class $N_CLASS \
                      --lr $LR \
                      --gamma $GAMMA \
                      --al-sampling-size $SAMPLING_SIZE \
                      --dataset-name $DNAME \
                      --output-dir $OUT_DIR \
                      --dataset-name $DNAME \
                      --max-eps $MAX_EPISODES \
                      --num-contact-subgraph $NUM_SUBGRAPH \
                      --min-density $MIN_DENSITY \
                      --max-density $MAX_DENSITY \
                      --threshold $THRESHOLD 
                      
