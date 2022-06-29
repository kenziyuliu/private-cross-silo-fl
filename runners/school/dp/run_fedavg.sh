#!/bin/bash
bash runners/school/run_fedavg.sh \
--example-dp \
--ex-clip 1 \
--ex-eps 1 \
--ex-delta 1e-3 \
$@
