#!/bin/bash

for i in $(seq -f "%02g" 30 30)
do
    echo "Testcase t$i"
    srun -N1 -n1 --gres=gpu:1 ./testsize /share/testcases/hw4/t$i ./output/t$i.out
    echo "----------"
done