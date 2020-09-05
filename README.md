# Correlated Semantic Hashing

PyTorch Implementation for paper "[Generative Semantic Hashing Enhanced via Boltzmann Machines]("https://www.aclweb.org/anthology/2020.acl-main.71")" (ACL'2020)


## Datasets

- Reuters
- 20NewsGroup
- TMC

We follow VDSH [(Chaidaroon and Fang, 2017)](https://arxiv.org/pdf/1708.03436.pdf) and use their pre-processed datasets, which can be found [here](https://github.com/unsuthee/VariationalDeepSemanticHashing/tree/master/dataset). After downloading these datasets, move them into the `dataset` directory.


## Running the script

To perform training and evaluation on our model, simply run:

```bash
python run.py
```

with the following supported arguments:

- `-g`(or `--gpunum`): to specify the GPU to train the model;
- `-d`(or `--dataset`): to specify the dataset, including `ng20`, `reuters`, `tmc`;
- `-b`(or `--nbits`): to specify the number of bits of hash codes;
- `--dropout`: Dropout probability (0 means no dropout);
- `-s`(or `--seed`): to set random seed;
- `--train_batch_size`: to specify the training batch size;
- ``--test_batch_size`: to specify the testing batch size;
- `-k`(or `--rank_k`): the number of ranks of perturbation for covariance;
- `-m`(or `--sample_m`): the number of samples used to construct the lower bound of ELBO;
- `--num_epochs`: number of epochs to run;
- `--lr`: learning rate.


## Citations

```
@inproceedings{zheng-etal-2020-generative,
    title = "Generative Semantic Hashing Enhanced via {B}oltzmann Machines",
    author = "Zheng, Lin  and
      Su, Qinliang  and
      Shen, Dinghan  and
      Chen, Changyou",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.71",
    doi = "10.18653/v1/2020.acl-main.71",
    pages = "777--788",
}
```

## License

This code is offered under the [MIT License](https://opensource.org/licenses/MIT).
