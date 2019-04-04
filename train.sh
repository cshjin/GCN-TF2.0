#!/bin/sh

ds=${1:-'citeseer'}
share=${2:-0.1}
for x in $(seq 1 20)
do
  python test_GCN.py --dataset=$ds --train_share=$share >> output_gcn
done