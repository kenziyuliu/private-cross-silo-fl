#!/bin/bash
bash runners/adni/main.sh \
--example-dp \
--ex-clip 0.5 \
--ex-eps 3 \
--ex-delta 1e-4 \
$@
