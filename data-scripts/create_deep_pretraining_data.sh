# Input files/directory (required)
REPO="/data/hulab/junjieh/deep"
SPM="${REPO}/models/mbart.cc25.v2/sentence.bpe.model"
DICT="${REPO}/models/mbart.cc25.v2/dict.txt"
ENTITY_FILE="${REPO}/data/wikidata/items.en"

#langs=( "ru" "uk" "ne" )
#LANGS=( "ru_RU" "uk_XX" "ne_XX" )

langs=( "uk" )
LANGS=( "uk_XX" )

for i in "${!langs[@]}"; do

    lang=${langs[i]}
    LANG=${LANGS[i]}
    echo "$lang $LANG"

    # Input text directory
    INPUTDIR="${REPO}/tools/sling/local/data/e/wiki/${lang}/"
    # Output text and binary directories
    OUTDIR="${REPO}/data/Wikipedia/wiki-max512-deep-spm250000/${LANG}/"
    DESTDIR="${REPO}/data/Wikipedia/wiki-max512-deep-spm250000-bin/${LANG}/"
    mkdir -p $OUTDIR $DESTDIR

    ## (1) Create 10 shards of $OUTDIR/train-{i}.{src,tgt,ctx,idx}, for i in [0,10).
    # Notes: Running each python program takes 25GB RAM. To save memory, we run two processes to preprocess [0-4] shards and [5-9] shards in parallel, which takes ~50GB RAM.
    # You need to install python wrappers of SLING, sentencepiece, tqdm.
    python $REPO/data-scripts/create_deep_pretraining_data.py --shard=0 --spm=$SPM --max_len=512 --outdir=$OUTDIR --entity_file=$ENTITY_FILE --inputdir=$INPUTDIR --rec_prefix="documents" --next_n_shards=5 &
    python $REPO/data-scripts/create_deep_pretraining_data.py --shard=5 --spm=$SPM --max_len=512 --outdir=$OUTDIR --entity_file=$ENTITY_FILE --inputdir=$INPUTDIR --rec_prefix="documents" --next_n_shards=5
    wait

    ## (2) Use the first 2000 lines of the first shard as the validation set.
    if [ ! -f $OUTDIR/train-0.src ]; then wait; fi
    if [ ! -f $OUTDIR/valid.src ]; then
        for s in 'src' 'tgt' 'qid' 'idx'; do
            first_shard="$OUTDIR/train-0.${s}"
            head -n 2000 $first_shard >> $OUTDIR/valid.${s}
            # Remove the first 2000 lines from the original first shard file.
            num_lines=$(wc -l "$first_shard" | cut -d ' ' -f 1)
            echo "$num_lines"
            sed -n 2001,${num_lines}p < $first_shard > $OUTDIR/train-${s}
            mv $OUTDIR/train-${s} $first_shard
        done
    fi

    ## (3) Binarize all shards into train-{i}.{src,tgt}.{bin,idx}
    # Notes: You need to install fairseq first.
    # Rename the files
    for f in $OUTDIR/*.src; do mv $f ${f/src/en_XX}; done
    for f in $OUTDIR/*.tgt; do mv $f ${f/tgt/${LANG}}; done
    fairseq-preprocess --validpref ${OUTDIR}/valid --source-lang en_XX --target-lang $LANG --srcdict ${DICT} --joined-dictionary --destdir ${DESTDIR} --workers 8 
    for i in 0 1 2 3 4 5 6 7 8 9; do
        rm -rf $DESTDIR/dict*
        fairseq-preprocess \
        --source-lang en_XX --target-lang $LANG \
        --trainpref $OUTDIR/train-${i} \
        --destdir $DESTDIR \
        --srcdict $DICT \
        --joined-dictionary \
        --workers 16
        # Rename the binarized training files.
        for f in $DESTDIR/train.en_XX-${LANG}*; do mv $f ${f/train/train-${i}}; done
    done

done
