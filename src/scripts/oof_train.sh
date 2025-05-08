#!/bin/bash

for ii in {0..4}
do
  echo "Running fold $ii..."
  python src/scripts/train.py data.dataset_args.val_fold=$ii
done
