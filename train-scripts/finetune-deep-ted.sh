export CUDA_VISIBLE_DEVICES=$1


REPO=/data/hulab/junjieh/deep
bin_dir=$REPO/data/ted-bin/en_XX-uk_XX/
tl='uk_XX'
sl='en_XX'
V=250000
MODEL=$REPO/models/mbart.cc25.v2/sentence.bpe.model
DICT=$REPO/models/mbart.cc25.v2/dict.txt
langs='uk_XX'

# Train a transformer with 512/2048 dimension size
# Let's use mbart_small defined in deep/fairseq/fairseq/models/bart/model.py
arch='mbart_small'
MAXLEN=512

## Note: Uncomment the following to try 3 baselines.
# ## 1. Initialized by random checkpoint
# PRETRAIN='None'
# save_dir=$REPO/outputs/finetune/ted-enuk/mbart-${langs}-random-uf32-mt2000-epoch5/

# ## 2. Initialized by DAE checkpoint
# PRETRAIN=$REPO/outputs/pretrain/mbart-${langs}-batch${BS}x8-wiki-max${MAXL}-dae-spm250000/checkpoint_44_50000.pt
# save_dir=$REPO/outputs/finetune/ted-enuk/mbart-${langs}-dae-uf32-mt2000-epoch5/

## 3. Initialized by DEEP checkpoint
PRETRAIN=$REPO/outputs/pretrain/mbart-${langs}-batch${BS}x8-wiki-max${MAXL}-deep-spm250000/checkpoint_44_50000.pt
save_dir=$REPO/outputs/finetune/ted-enuk/mbart-${langs}-deep-uf32-mt2000-epoch5/


mkdir -p $save_dir/runs
echo "save_dir=$save_dir"
python $REPO/fairseq/fairseq_cli/train.py $bin_dir \
    --save-dir $save_dir \
    --task translation_from_pretrained_bart \
    --arch $arch \
    --restore-file $PRETRAIN \
    --source-lang ${sl} --target-lang ${tl} \
    --encoder-normalize-before --decoder-normalize-before \
    --label-smoothing 0.2 --criterion label_smoothed_cross_entropy \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 \
    --fp16 \
    --use_gpu \
    --langs $langs \
    --use-lang-tokens \
    --save-interval 1 --save-interval-updates 1000 \
    --min-loss-scale 0.0001 \
    --all-gather-list-size 16384 \
    --max-tokens-valid 1024 \
    --max-tokens 1024 \
    --update-freq 32 \
    --max-source-positions ${MAXLEN} \
    --max-target-positions ${MAXLEN} \
    --required-batch-size-multiple 8 \
    --clip-norm 25 \
    --lr 3e-5 --min-lr -1 \
    --weight-decay 0.0 \
    --warmup-updates 2500 \
    --skip-invalid-size-inputs-valid-test \
    --lr-scheduler polynomial_decay \
    --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.0 --activation-dropout 0.0 \
    --patience -1 \
    --max-epoch 0  \
    --no-epoch-checkpoints \
    --max-update 10000 \
    --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
    --total-num-update 10000 1> $save_dir/log 2> $save_dir/err
