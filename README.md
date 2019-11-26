# GCN-TF2.0

This is the vanilla implementation of graph convolution networks (GCN) with tensorflow 2.x.

Instead of the Keras API, this implementation take the GradientTape as the optimization process.

No placeholder, no tf.app, tf.flags, and some APIs deprecated after TF 2.x.

Reference: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907).

Parts of the code coming from the author's original implementation [code](https://github.com/tkipf/gcn). 

Credits to Thomas Kipf. Thanks.

## usage

`python train.py [options]`

You can specify the options as
```
  --dataset, choosing from 'citeseer', 'cora', 'cora_ml', 'pubmed', 'polblogs' and 'dblp'
```
Check out the `train.py` for more detailed arguments.

## LICENSE
The project is under [MIT](./LICENSE) license.
