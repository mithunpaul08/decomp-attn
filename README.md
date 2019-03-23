# Decomposable Attention Model for Sentence Pair Classification

Implementation of the paper [A Decomposable Attention Model for Natural Language Inference](https://arxiv.org/abs/1606.01933). Parikh et al. EMNLP 2016.

The same model can be used for generic sentence pair classification tasks (e.g. paraphrase detection), in addition to natural language inference.

## Data

note: the harvard code expects python 2.7 only.

```
conda create --name harvardSnli python=2
conda activate harvardSnli
pip install numpy tqdm
```
note to self:



## Preprocessing
- First segregate the FEVER data into a file full of claims, evidences and labels alone. Do this for both dev and train.
```
python process-fever.py --data_folder data-local --out_folder outputs
```

- Next download glove from [here](http://nlp.stanford.edu/projects/glove/)

**Note: you might need atleast 2 GB for both these data sets.**

Have installed pytorch on this environment using :

`conda install pytorch torchvision cudatoolkit=8.0 -c pytorch`

There is a conda environment in clara called : `py2_decompattn_nonallennlp`

after downloading  glove these are the two commands i ran in harvard code :

```
python preprocess.py --srcfile outputs/src-train.txt --targetfile outputs/targ-train.txt --labelfile outputs/label-train.txt --srcvalfile outputs/src-dev.txt --targetvalfile outputs/targ-dev.txt --labelvalfile outputs/label-dev.txt --outputfile outputs/hdf5 --glove data-local/glove.840B.300d.txt
```

```
python get_pretrain_vecs.py --dictionary outputs/hdf5.word.dict --glove data/glove/glove.840B.300d.txt --outputfile outputs/glove.hdf5
```


Then run:
```
python preprocess-entail.py --srcfile path-to-sent1-train --targetfile path-to-sent2-train
--labelfile path-to-label-train --srcvalfile path-to-sent1-val --targetvalfile path-to-sent2-val
--labelvalfile path-to-label-val --srctestfile path-to-sent1-test --targettestfile path-to-sent2-test
--labeltestfile path-to-label-test --outputfile data/entail --glove path-to-glove
```
Here `path-to-sent1-train` is the path to the `src-train.txt` file created from running `process-snli.py` (and `path-to-sent2-train` = `targ-train.txt`, `path-to-label-train` = `label-train.txt`, etc.)

`preprocess-entail.py` will create the data hdf5 files. Vocabulary is based on the pretrained Glove embeddings,
with `path-to-glove` being the path to the pretrained Glove word vecs (i.e. the `glove.840B.300d.txt`
file).

For SNLI `sent1` is the premise and `sent2` is the hypothesis.

Now run:
```
python get_pretrain_vecs.py --glove path-to-glove --outputfile data/glove.hdf5
--dictionary path-to-dict
```
`path-to-dict` is the `*.word.dict` file created from running `preprocess.py`

## Training
To train the model, run 
```
th train.lua -data_file path-to-train -val_data_file path-to-val -test_data_file path-to-test
-pre_word_vecs path-to-word-vecs
```
Here `path-to-word-vecs` is the hdf5 file created from running `get_pretrain_vecs.py`.

You can add `-gpuid 1` to use the (first) GPU.

The model essentially replicates the results of Parikh et al. (2016). The main difference is that
they use asynchronous updates, while this code uses synchronous updates.

## Predicting
To predict on new data, run
```
th predict.lua -sent1_file path-to-sent1 -sent2_file path-to-sent2 -model path-to-model
-word_dict path-to-word-dict -label_dict path-to-label-dict -output_file pred.txt
```
This will output the predictions to `pred.txt`. `path-to-word-dict` and `path-to-label-dict` are the
*.dict files created from running `preprocess.py`

## Contact

Written and maintained by <a href="http://yoon.io">Yoon Kim</a>.

## Licence
MIT
