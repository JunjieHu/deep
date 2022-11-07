#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
BASE=$HOME
REPO=/data/hulab/junjieh/deep/
sl='en_XX'
tl='uk_XX'
SL='en'
TL='uk'


bin_dir=$REPO/data/ted-bin/en_XX-uk_XX/
detok_ref=$REPO/data/ted/test.uk

langs="uk_XX"
# Checkpoint folder
mt_dir=$REPO/outputs/finetune/ted-enuk/mbart-${langs}-deep-uf32-mt2000-epoch5/
cd $mt_dir

ckpt=checkpoint_best.pt
path=$mt_dir/$ckpt
mt=${ckpt/pt/mt}
out_file=$mt_dir/${ckpt/pt/mt}
echo $ckpt $mt $bin_dir $mt_dir

fairseq-generate \
    $bin_dir \
    --task translation_from_pretrained_bart_tag \
    --tasks "mt,dae" \
    --source-lang ${sl} --target-lang ${tl} \
    --use-lang-tokens \
    --langs $langs \
    --path $path \
    --beam 5 --lenpen 1.2 \
    --gen-subset test \
    --batch-size 16 \
    --use_gpu \
    --skip-invalid-size-inputs-valid-test \
    --scoring sacrebleu \
    --remove-bpe="sentencepiece" > $out_file

# Tokenized test
REPLACE_UNICODE_PUNCT=$REPO/tools/mosesdecoder/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$REPO/tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$REPO/tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl
REMOVE_DIACRITICS=$REPO/tools/mosesdecoder/scripts/tokenizer/remove-diacritics.py
NORMALIZE_ROMANIAN=$REPO/tools/mosesdecoder/scripts/tokenizer/normalise-romanian.py
TOKENIZER=$REPO/tools/mosesdecoder/scripts/tokenizer/tokenizer.perl

sys=$mt_dir/${mt}.hyp
ref=$mt_dir/${sl}_${tl}.ref
cat $out_file | grep -P "^H" | sort -V | cut -f 3- > $sys
cat $out_file | grep -P "^T" | sort -V | cut -f 2- > $ref

for file in $sys $ref; do
  cat $file \
  | $REPLACE_UNICODE_PUNCT \
  | $NORM_PUNC -l $TL \
  | $REM_NON_PRINT_CHAR \
  | python $REMOVE_DIACRITICS \
  | $TOKENIZER -no-escape -l $TL \
  > ${file}.tok
done
echo "sacrebleu -tok none -s none -b ${ref}.tok < ${sys}.tok"
sacrebleu -tok none -s none -b ${ref}.tok < ${sys}.tok
