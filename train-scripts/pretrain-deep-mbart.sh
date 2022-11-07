REPO=/data/hulab/junjieh/deep
MAXL=512
DATA=$REPO/data/Wikipedia/wiki-max${MAXL}-deep-spm250000-bin/
SAVE=$REPO/outputs/

## Pre-train on uk_XX
langs="uk_XX"
BS='8x512'
OUTDIR=$SAVE/pretrain/mbart-${langs}-batch${BS}x8-wiki-max${MAXL}-deep-spm250000

mkdir -p $OUTDIR/log $OUTDIR/save

# We use a while-loop to load the latest checkpoint to continue training 
# if the job is terminated before the max step.
while : 
do
python $REPO/fairseq/train-tpu.py \
$DATA \
--task=multilingual_context_shard_denoising \
--tensorboard-logdir=$OUTDIR/log \
--arch=mbart_small \
--attention-dropout=0.0 \
--relu-dropout=0.0 \
--adaptive-softmax-dropout=0.0 \
--no-progress-bar \
--criterion=cross_entropy \
--lr-scheduler=polynomial_decay \
--min-lr=-1 \
--skip-invalid-size-inputs-valid-test \
--optimizer=adam \
--adam-betas="(0.9, 0.999)" \
--lr=[0.0003] \
--end-learning-rate=0.0 \
--warmup-updates=10000 \
--share-decoder-input-output-embed \
--dropout=0.0 \
--weight-decay=0.01 \
--train-subset=train \
--valid-subset=valid \
--max-update=500000 \
--save-dir=$OUTDIR/ \
--restore-file=$OUTDIR/checkpoint_last.pt \
--mask=0.3 \
--mask-random=0.1 \
--poisson-lambda=3.5 \
--permute-sentences=1 \
--mask-length=span-poisson \
--replace-length=1 \
--encoder-normalize-before \
--decoder-normalize-before \
--max-source-positions=$MAXL \
--max-target-positions=$MAXL \
--share-all-embeddings \
--layernorm-embedding \
--log_steps=100 \
--log-format=json \
--seed=1111 \
--min-loss-scale=0.0001 \
--model-parallel-size=1 \
--required-batch-size-multiple=1 \
--validate-interval-updates=-1 \
--validate-interval=5 \
--bucket-cap-mb=25 \
--clip-norm=0.1 \
--optimizer-overrides={} \
--save-interval-updates=5000 \
--keep-interval-updates=10 \
--best-checkpoint-metric=loss \
--no-epoch-checkpoints \
--patience=-1 \
--adam-eps=1e-06 \
--power=1 \
--langs=$langs \
--add-lang-token \
--total-num-update=500000 \
--num-workers=8 \
--input_shapes ${BS} \
--update-freq=[1] \
--no-progress-bar \
--bf16 \
--num-shards 10 \
--bpe sentencepiece \
--multilang-sampling-alpha 0.7 \
--save_specific_checkpoints "5000,50000,100000,150000,200000,300000,400000,500000" \
--sentencepiece-model $REPO/models/mbart.cc25.v2/sentence.bpe.model # 2> $OUTDIR/err 

sleep 1
done

