#!/bin/bash
bash runners/gleam/dp/main.sh \
--trainer ifca_mrmtl \
--num-clusters 3 \
--lambda 0.001 \
$@
