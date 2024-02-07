#!/bin/bash

m="map_auc_g_both"
y="mavg_bulbf_ccaf"
task="both"

for m in sbp_reactivity_both sbp_auc_i_both sbp_auc_g_both
do

    case_dir="Y-${y}_M-${m}_task-${task}"
    for model in ridge lasso
    do
    	output_dir="results/${case_dir}/predictions/${model}"
    	cmd="python 02-predictions.py --target ${y} --mediator ${m} --task ${task} --model ${model} --output_dir ${output_dir}"
        echo "running ${cmd}"
    	eval $cmd
    done
done
