#!/bin/bash
bash runners/gleam/dp/main.sh \
--trainer ifca_finetune \
--num-clusters 3 \
$@
