#!/bin/bash
bash runners/gleam/dp/main.sh \
--trainer mocha \
--mocha_mode primal \
--lambda 0.001 \
$@
