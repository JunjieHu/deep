# DEEP: DEnoising Entity Pre-training for Neural Machine Translation


## Installation
Here are a list of important tools for installation. We also provide a conda env file [py39_env.txt](./py39_env.yml).
- SLING (Please see [SLING.md](./SLING.md) for more details.)
- Python: sentencepiece, tqdm. 
- [Our modified fairseq for TPU/GPU](./fairseq/)
```
cd fairseq
pip install --editable ./
```

## Download
- Download the [mbart.CC25](https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.cc25.v2.tar.gz) checkpoints 
- Download the preprocessed data from this [Google drive folder](https://drive.google.com/drive/folders/15Wim7cR85jz1UGLBNEpgf-FPdgJFLWO5?usp=share_link): 
  - [English entity items](https://drive.google.com/file/d/14cZ8clBAFobcxmw_eWZAUTh19TVFEFX8/view?usp=share_link) 
  - [Wikipedia text with code-switched entities for pretraining](https://drive.google.com/drive/folders/1Wq7_oRDcRpPG8QqQh_UFmX0bCsv42au9?usp=share_link)
  - [Ted talk En-Uk translation for finetuning](https://drive.google.com/drive/folders/1nrP1-KCcrvGndloCwZkFXe7hD-4n8KQq?usp=share_link)

## Perform SLING's entity linking
After installing SLING, run the following to perform entity linking on Wikipedia article.
```
lang=$1
version=20221101  # the version we used
./run.sh --download_wikidata --download_wikipedia --wikipedia $version --language $lang
```

## Prepare DEEP's Pre-training Data
After the installation of above tools, run the following to create DEEP's pre-training data. 
```
bash data-scripts/create_deep_pretraining_data.sh
```
This will generate two folders. Each language (e.g., uk_XX) has its sub-folder:
- `data/Wikipedia/wiki-max512-deep-spm250000/uk_XX/` - Raw text : train-{0-9}.{en_XX,uk_XX,idx,qid}, and valid.{en_XX,uk_XX,idx,qid}
- `data/Wikipedia/wiki-max512-deep-spm250000-bin/uk_XX` - Fairseq's binarized data: train-{0-9}.en_XX-uk_XX.{en_XX,uk_XX}.{bin,idx}

## Pre-training on TPU
We pre-train the mBART models using TPU on Google Cloud Platform. The model is pre-trained on the pre-training data created above. We modify the Fairseq's repository such that we can run the code on GCP's TPU.
```
bash train-scripts/pretrain-deep-mbart.sh
```

## Finetune on Downstream MT Task
Here we give an example of fune-tuning our pre-trained models on the Ted En-Uk dataset. Replace [GPU ID] by an integer (e.g., 0, 1, ...) indicating which GPU to use.
```
bash train-scripts/finetune-deep-ted.sh [GPU ID]  
```


## Evaluate on Downstream MT Task
```
bash train-scripts/test_ted_enuk_deep.sh [GPU ID]
```