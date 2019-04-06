# Vanilla GCN

This is the vanilla implementation of graph convolution networks (GCN).

Reference: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907).

Parts of the code coming from the author's original implementation [code](https://github.com/tkipf/gcn). 

Credits to Thomas Kipf. Thanks.

## usage

`python test_GCN.py [options]`

You can specify the options as
```
  --dataset, choosing from 'citeseer', 'cora', 'cora_ml', 'pubmed', 'polblogs'
  --train_share, the share used to training, [0.1, 0.8]
```
Check out the `test_GCN.py` for more detailed arguments.

## LICENSE
The project is under [MIT](./LICENSE) license.