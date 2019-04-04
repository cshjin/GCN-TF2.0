#!/bin/sh
# start from train_all, and it pass parameter to test.sh

for ds in 'citeseer' 'cora' 'cora_ml' 'pubmed' 'polblogs' 'dblp'
do 
  for share in $(seq 0.1 0.1 0.8)
  do
    sh train.sh $ds  $share &
  done
done