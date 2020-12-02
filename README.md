# test-bert

## How to Run

### Prerequisites

- Install [conda](https://docs.conda.io/projects/conda/en/4.6.1/user-guide/install/index.html)
- Create a new Python3.6 virtual env with the name `pytorch`

```bash
# create virtual env
conda create --name pytorch python=3.6
# activate virtual env
conda activate pytorch
# update all packages in the env
conda update --all 
```

- Install pytorch & transformers

```bash
# activate virtual env if not yet
conda activate pytorch
# install pytorch
conda install pytorch torchvision -c pytorch
# install transformers
pip install transformers[torch]
pip install pandas tabulate nltk
```

Verify Pytorch installation:

```bash
$ conda activate pytorch
$ python
>>> import torch 
>>> print(torch.__version__) 
1.7.0
>>>
```

### Run Test Program

```bash
python test.py
```

The program will ask you to input a sentence and return:
- a fact found 
- plus any other candidate found