testdata=$1
mfolder=$2

python src/test_3d.py \
    --net T3D \
    --folder ${testdata} \
    --mfolder ${mfolder} \
    --testtype 1 \
    --ntype 3d \
    --shape 40 40 16 \
    --mtype irregular \
    --mrate 0.5 \
    --urf 4 \
    --norm absmax \
    --moname recon_best