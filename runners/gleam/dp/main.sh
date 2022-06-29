#!/bin/bash
bash runners/gleam/main.sh \
--example-dp \
--ex-clip 6.0 \
--ex-eps 1 \
--ex-delta 1e-4 \
$@
