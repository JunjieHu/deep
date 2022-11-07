""" Extract source/target/mask indices/descriptions from next n shards of files, i.e.,
documents-0000{arg.shard + i}-of-00010.rec, where i in [i, i+n). For example, 
set `args.next_n_shards=5` and `args.shard=0`, this will process the next 5 shards
starting from 0-th shard, i.e., generating train-{i}.{src,tgt,idx,qid}, for i in [0,5).
"""
import logging
from typing import NamedTuple, Tuple, List
import sentencepiece as spm
import sling
import time
import sys
import os
from tqdm import tqdm
import pickle
import argparse
from collections import Counter
LOGGER = logging.getLogger(__name__)


# Global dictionaries
QID2SM = {} # dict: qid (str) -> spm-tokenized text (list of str)
QID2M = {}  # dict: qid (str) -> raw text (list of str)
QID2F = Counter()  # dict: qid (str) -> frequency (int)

# Some magic commands for SLING
commons = sling.Store()
DOCSCHEMA = sling.DocumentSchema(commons)
commons.freeze()


def get_qid2mention(file):
    """ Read a dictionary from WikiData's QID to the entity text and its description
    """
    start = time.time()
    skip = 0
    print('Reading {} file'.format(file))
    for l in open(file, 'r'):
        ls = l.strip().split('\t')
        if len(ls) != 3 or not ls[0].startswith('Q'):
            skip += 1
        else:
            QID2M[ls[0]] = (ls[2], ls[1])
            QID2F[ls[0]] = 0
    print('Reading {} entities, skip {}, time={}'.format(len(QID2M), skip, time.time() - start))


def get_mentions(document: sling.nlp.document.Document):
    """ Returns the string ID of the linked entity for this mention.
    Credit: Thanks Bhuwan for sharing the code.
    """
    mentions = document.mentions
    linked_mentions = []
    cnts = {i:0 for i in range(5)}
    for i, mention in enumerate(mentions):
        if "evokes" not in mention.frame or type(mention.frame["evokes"]) != sling.Frame:
            continue
        if "is" in mention.frame["evokes"]:
            if type(mention.frame["evokes"]["is"]) == sling.Frame:
                linked_mentions.append((mention.begin, mention.end, mention.frame["evokes"]["is"].id))
        else:
            if mention.frame["evokes"].id:
                linked_mentions.append((mention.begin, mention.end, mention.frame["evokes"].id))
    return linked_mentions


def get_idx2spmidx(toks, spm_toks):
    ''' Helper function: Build a dictionary to map the word token index to its first subword token index in a sentence
    '''
    MARK = '▁'
    idx2spmidx = {}
    idx = -1
    for spmidx, s in enumerate(spm_toks):
        if s.startswith(MARK):
            idx += 1
            idx2spmidx[idx] = [spmidx]
        else:
            idx2spmidx[idx].append(spmidx)

    # check
    for idx, t in enumerate(toks):
        spm_txt = ''.join(spm_toks[j] for j in idx2spmidx[idx])
        assert spm_txt.startswith(MARK), 'spm_txt={}'.format(spm_txt)
        # assert spm_txt[len(MARK):] == t, f'spm_txt[2:]={spm_txt[len(MARK):]}, t={t}, idx={idx}'
    return idx2spmidx


def get_mention_tokens_descriptions(mentions, tokenizer):
    spm_mentions = []
    time_cnt = {'qid':0, 'token': 0}
    skip_cnt = 0
    for (bidx, eidx, qid) in mentions:
        if qid not in QID2M:
            skip_cnt += 1
            continue
        try:
            if qid not in QID2SM:
                mdes, mtxt = QID2M[qid]
                mdes = ' '.join(mdes.split(' ')[:10]) # truncated to first 10 words
                spm_toks_mdes = tokenizer(mdes)  # list
                spm_toks_mtxt = tokenizer(mtxt)
                QID2SM[qid] = (spm_toks_mdes, spm_toks_mtxt)
            else:
                (spm_toks_mdes, spm_toks_mtxt) = QID2SM[qid]
            spm_mentions.append((bidx, eidx, spm_toks_mdes, spm_toks_mtxt, qid))
        except:
            print(' Qid={} not found'.format(qid))
    return spm_mentions


def create_src_tgt_mask_description(args, spm_mentions, spm_toks, toks, idx2spmidx):
    MAXL = args.max_len
    def _prepare_record(src, tgt, mask, des, qids):
        # Pad the src/tgt to the max sequence length
        # nsrc = src + ['</s>'] + ['<pad>'] * (MAXL - len(src)) if len(src) <= MAXL else src[0: MAXL] + ['</s>']
        # ntgt = tgt + ['</s>'] + ['<pad>'] * (MAXL - len(tgt)) if len(tgt) <= MAXL else tgt[0: MAXL] + ['</s>']
        nsrc = ['[dae]'] + src + ['</s>'] if len(src) <= MAXL else ['[dae]'] + src[0: MAXL] + ['</s>']
        ntgt = tgt + ['</s>'] if len(tgt) <= MAXL else tgt[0: MAXL] + ['</s>']
        return (nsrc, ntgt, mask, des, qids)

    idx2spm_mentions = {m[0]: m for m in spm_mentions}
    src, tgt, mask, des, qids = [], [], [], [], []
    records = []
    idx = 0
    while idx < len(toks):
        bspm = idx2spmidx[idx][0]
        # Check if the tok[idx] is the begining of a mention
        if idx in idx2spm_mentions:
            bidx, eidx, spm_toks_mdes, spm_toks_mtxt, qid = idx2spm_mentions[idx]
            assert idx == bidx, 'idx={}, bidx={}'.format(idx, bidx)
            espm = idx2spmidx[eidx - 1][-1] + 1
            # Check if adding the current tokenized mention to the src or tgt
            # would exceed the max sequence length.
            if (len(spm_toks_mtxt) + len(src) > MAXL or (espm - bspm) + len(tgt) > MAXL):
                record = _prepare_record(src, tgt, mask, des, qids)
                records.append(record)
                src, tgt, mask, des, qids = [], [], [], [], []
            # Add the mention words to src/tgt/des/mask
            mask.extend([len(src) + i for i in range(len(spm_toks_mtxt))])
            src.extend(spm_toks_mtxt)  # src add spm_toks_mtxt
            tgt.extend(spm_toks[bspm: espm])
            des.extend(['<s>'] + spm_toks_mdes)
            qids.append(qid)
            QID2F[qid] += 1
            idx = eidx
        else:
            espm = idx2spmidx[idx][-1] + 1
            # Check if adding the current tokenized word to the src or tgt
            # would exceed the max sequence length
            if (espm - bspm) + len(src) > MAXL or (espm - bspm) + len(tgt) > MAXL:
                record = _prepare_record(src, tgt, mask, des, qids)
                records.append(record)
                src, tgt, mask, des = [], [], [], []
            src.extend(spm_toks[bspm: espm])
            tgt.extend(spm_toks[bspm: espm])
            idx += 1
    # If the last tgt sequence is too short, just don't add it to the records
    if 0.9 * MAXL < len(tgt) <= MAXL:
        record = _prepare_record(src, tgt, mask, des, qids)
        records.append(record)
    return records


def init_processor(args):
    # Load SPM tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(args.spm)
    tokenizer = tokenizer.EncodeAsPieces
    return tokenizer


def init_file_writer(shard):
    # Initilize `args.nshards` writing handlers for src/tgt/idx/ctx
    fsrc = open(os.path.join(args.outdir, 'train-{}.src'.format(shard)), 'w')
    ftgt = open(os.path.join(args.outdir, 'train-{}.tgt'.format(shard)), 'w')
    fidx = open(os.path.join(args.outdir, 'train-{}.idx'.format(shard)), 'w')
    fctx = open(os.path.join(args.outdir, 'train-{}.qid'.format(shard)), 'w')
    return fsrc, ftgt, fidx, fctx


def main(args):
    # Reading all extracted English entity takes time and 25GB RAM.
    get_qid2mention(args.entity_file)
    # Process the next n shards starting from the `args.shard` shard.
    for s in range(args.next_n_shards):
        print('start reading {} shard'.format(args.shard + s))
        extract_one_shard_text(args, args.shard + s)
    # Save the qid-frequency directory
    if args.qid_path is not None:
        with open(args.qid_path, 'w') as fout:
            for qid, freq in QID2F.most_common():
                fout.write('{}\t{}\n'.format(qid, freq))

def extract_one_shard_text(args, shard):
    # Initialize the SPM tokenizer
    tokenizer = init_processor(args)
    
    # Set file path
    rec_file = os.path.join(args.inputdir, '{}-0000{}-of-00010.rec'.format(args.rec_prefix, shard))
    total_num = sum(1 for _ in sling.RecordReader(rec_file))
    fsrc, ftgt, fidx, fctx = init_file_writer(shard)
    print('- Begin reading {} file.'.format(rec_file))

    # Iterate each doc in the ref_file.
    start = time.time()
    spm_cnt = men_cnt = 0
    total_record_cnt = 0
    for k, rec in tqdm(sling.RecordReader(rec_file), total=total_num):
        test = time.time()

        # Load a doc by mapping the content into document schema
        store = sling.Store(commons)
        doc = sling.Document(store.parse(rec), store, DOCSCHEMA)
        if len(doc.tokens) < 0.8 * args.max_len: # skip if the doc is too short
            continue
        mentions = get_mentions(doc)

        # Tokenize doc's text & reindexing toks' indices to spm_toks' indices
        toks = [t.word.strip() for t in doc.tokens] # list of strings
        txt = ' '.join(toks)  # string
        spm_toks = tokenizer(txt)  # list of strings
        try:
            idx2spmidx = get_idx2spmidx(toks, spm_toks) # dict: int->list
        except:
            print(' idx2spmidx error')
            continue

        # Get mention's description & text, and tokenize
        spm_mentions = get_mention_tokens_descriptions(mentions, tokenizer)
        men_cnt += len(mentions)
        spm_cnt += len(spm_mentions)

        # Prepare the records
        records = create_src_tgt_mask_description(args, spm_mentions, spm_toks, toks, idx2spmidx)
        total_record_cnt += len(records)
        for (s, t, m, d, q) in records:
            fsrc.write(' '.join(s) + '\n')
            ftgt.write(' '.join(t) + '\n')
            fidx.write(' '.join(str(midx) for midx in m) + '\n')
            fctx.write(' '.join(q) + '\n')
    print('Finish processing {} records'.format(total_record_cnt))

    # Close all writers
    for f in [fsrc, ftgt, fidx, fctx]:
        f.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='create deep pretrain data')
    parser.add_argument('--outdir', type=str, default=None, required=True, help='the output directory')
    parser.add_argument('--shard', type=int, default=0, required=True, help='start index of a shard')
    parser.add_argument('--spm', type=str, default=None, required=True)
    parser.add_argument('--max_len', type=int, default=512, required=True)
    parser.add_argument('--entity_file', type=str, default=None, required=True)
    parser.add_argument('--inputdir', type=str, default=None, required=True)
    parser.add_argument('--next_n_shards', type=int, default=5)
    parser.add_argument('--qid_path', type=str, default=None)
    parser.add_argument('--rec_prefix', type=str, default='documents')
    args = parser.parse_args()
    # Reserve two tokens for [dae] and </s>.
    args.max_len = args.max_len - 2

    main(args)




