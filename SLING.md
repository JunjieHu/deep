# Work with SLING for entity linking

## Install requirements

On a *Linux* machine, clone the sling repository and follow [this script](https://github.com/google/sling/blob/master/setup.sh).
Before running the following, change the `PYVER` to a preferred version (e.g., `PYVER=3.9`).
It requires `sudo` access to install necessary libraries below.
```
cd tools
git clone https://github.com/google/sling.git
cd sling
./setup.sh
```

## Build SLING tools

After Bazel is installed, we are ready to build SLING. Follow [here](https://github.com/google/sling/blob/master/doc/guide/install.md#building) and run
```
./tools/buildall.sh
```

Then install the Python API:
```
# In whichever virtual environment 
pip install http://www.jbox.dk/sling/sling-2.0.0-py3-none-linux_x86_64.whl
```

## Build Wikipedia & Wikidata database

To run the entity linker on a language, we first need to download and "build" a SLING database (there may be a more appropriate term).
First, [download the dump](https://github.com/google/sling/blob/master/doc/guide/wikiflow.md#download-dumps):
```sh
lang=$1
version=20200901  # the version we used
./run.sh --download_wikidata --download_wikipedia --wikipedia $version --language $lang
```

Then, construct a file called `idf.repo` which is used to find entities and link them to Wikidata on a new text.

```sh
# Specify a TMPDIR, where sling stores intermediate temporary files 
# Caution: Disk usage
export TMPDIR=/path/to/large/disk/space
mkdir -p $TMPDIR

# --build_idf generates a file "idf.repo"
# --silver_annotation will annotate wikipedia text.
./run.sh --build_wiki --build_idf --silver_annotation --language $lang
```

## Apply SLING to a new text

Please download [mkcorpus.py](https://gist.github.com/rooa/efcb5d0dd67ec9554446b0b835bbdc47) and [annotate.py](https://gist.github.com/rooa/0422373473aa92068c4504cebfae800a) and store them somewhere.

```sh
# before running, change the variables accordingly
LANG=en 
WORKDIR=/location/of/downloaded/scripts
SLINGDIR=/location/of/cloned/sling/repo
SLING_PYTHON_DIR=/path/to/python/with/sling/api

# path to the file to annotate - format is plain text
INPUT=/path/to/text/file

# Force use the python containing sling installation
export PATH=$SLING_PYTHON_DIR:$PATH

# has to be at the root directory of sling repository
pushd $SLINGDIR

# format the text into sling corpus
python $WORKDIR/mkcorpus.py --input $INPUT --output $INPUT.sling_corpus

# run entity recognition and linking
python $WORKDIR/annotate.py \
    --annotate_input $INPUT.sling_corpus \
    --annotate_output $INPUT.annotated.rec \
    --language $LANG

echo "Stored the annotation output at $INPUT.annotated.rec."

popd
```

## Building an entity linker for a new language

SLING by default only supports [certain languages](https://github.com/google/sling/issues/438#issuecomment-579159616). Please take a look at the link on what to do.

Basically, adapting to a new language (whose Wikipedia dump exists) requires:
- Add language code to the array of languages in `sling/nlp/wiki/wiki.cc`
- Add language prefixes for image and category in that language in `sling/nlp/wiki/wiki-parser.cc`
- Add a new template file at `data/wiki/uk/templates.sling` (<- Ukrainian example)
  - NOTE: make a directory with two-char language code, then place a `templates.sling` file
  
Example template.sling for Ukrainian:
```
{=/wp/templates/uk

"!":      {type: "text" text: "|"}
"!!":     {type: "text" text: "||"}
")!":     {type: "text" text: "]"}
"!((":    {type: "text" text: "[["}
"))!":    {type: "text" text: "]]"}
"(":      {type: "text" text: "{"}
")":      {type: "text" text: "}"}
"((":     {type: "text" text: "{{"}
"))":     {type: "text" text: "}}"}
"(((":    {type: "text" text: "{{{"}
")))":    {type: "text" text: "}}}"}
"·":      {type: "text" text: "&nbsp;<b>&middot;</b>"}
"•":      {type: "text" text: "&nbsp;&bull;"}
"=":      {type: "text" text: "="}

}
```

After all the edits, re-compile SLING with `./tools/buildall.sh`. SLING is now ready for processing a new language. Then,

- Get the Wikipedia dump: `./run.sh --download_wikipedia --wikipedia [TIMESTAMP] --language [LANG]`
- Build `idf.repo`: `./run.sh --build_wiki --build_idf --silver_annotation --language [LANG]`

After these steps, it should be ready to apply entity linking on the new language using the same process as [above](#apply-sling-to-a-new-text).