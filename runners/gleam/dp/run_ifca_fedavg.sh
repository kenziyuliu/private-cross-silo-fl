#!/bin/bash
bash runners/gleam/dp/main.sh \
--trainer ifca_fedavg \
--num-clusters 3 \
$@
