testdata=$1
mfolder=$2

python src/test_2d.py \
    --net T2D \
    --folder ${testdata} \
    --mfolder ${mfolder} \
    --testtype 1 \
    --shape 40 40 \
    --mtype irregular \
    --mrate 0.5 \
    --urf 4 \
    --norm absmax \
    --moname recon_best \
    --ordered True