#!/bin/bash
bash runners/rotated_mnist/main.sh \
--example-dp \
--ex-clip 1.0 \
--ex-eps 1 \
$@
