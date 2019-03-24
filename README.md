my fork of yoon kim's DA [model](https://github.com/harvardnlp/decomp-attn). I use it for FEVER data

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
note to self:- There is a conda environment in clara called : `py2_decompattn_nonallennlp`



## Preprocessing
- First segregate the FEVER data into a file full of claims, evidences and labels alone. Do this for both dev and train.

    - note: output of this command will be in .txt format and in the `outputs/` folder
```
python process-fever.py --data_folder data-local --out_folder outputs
```

- Next download glove from [here](http://nlp.stanford.edu/projects/glove/) and unzip it into the data folder.

    - **Note: you might need atleast 2 GB for both these data sets.**

    

after downloading  glove these are the  commands i ran in harvard code :

- to convert all your train and dev files to hdf5 format run this command:
- you have to run it at the level which had README.md
```
python preprocess.py --srcfile outputs/src-train.txt --targetfile outputs/targ-train.txt --labelfile outputs/label-train.txt --srcvalfile outputs/src-dev.txt --targetvalfile outputs/targ-dev.txt --labelvalfile outputs/label-dev.txt --outputfile outputs/hdf5 --glove data-local/glove.840B.300d.txt
```
- to convert the words in your vocabulary to its embeddings and then into hdf5 format
```
python get_pretrain_vecs.py --dictionary outputs/hdf5.word.dict --glove data-local/glove.840B.300d.txt --outputfile outputs/glove.hdf5
```

