#!/bin/bash

y="mavg_bulbf_ccaf"
task="both"
n_boots=5000

model="ridge"
for m in sbp_reactivity_both
do
    case_dir="Y-${y}_M-${m}_task-${task}"
	output_dir="results/${case_dir}/bootstrapping/${model}"
	cmd="python bootstrapping_parallel.py --target ${y} --mediator ${m} --task ${task} --n_boots ${n_boots} --output_dir ${output_dir}"
	eval $cmd
done

