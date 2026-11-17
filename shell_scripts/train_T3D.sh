dsname=$1

python ./src/train.py \
    --net T3D \
    --dpath ${dsname} \
    --gpunum 1 \
    --batchSize 1 \
    --dtype 3d \
    --epoch 20
