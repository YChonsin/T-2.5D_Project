dsname=$1

python ./src/train.py \
    --net T2D \
    --dpath ${dsname} \
    --gpunum 1 \
    --batchSize 1 \
    --dtype 2d \
    --epoch 20
