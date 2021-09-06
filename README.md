# Contra-OOD

Code for EMNLP 2021 paper [Contrastive Out-of-Distribution Detection for Pretrained Transformers](https://arxiv.org/abs/2104.08812).

## Requirements
* [PyTorch](http://pytorch.org/)
* [Transformers](https://github.com/huggingface/transformers)
* datasets
* wandb
* tqdm
* scikit-learn

## Dataset
Most datasets used in paper are automatically downloaded by the datasets package. Instructions on downloading sst2 and multi30k are provided in ``readme.txt`` under the ``data`` folder.

## Training and Evaluation

Finetune the PLM with the following command:

```bash
>> python run.py --task_name task
```

The ``task_name`` parameter can take ``sst2``, ``imdb``, ``trec``, or ``20ng``. The training loss and evaluation results on the test set are synced to the wandb dashboard.
