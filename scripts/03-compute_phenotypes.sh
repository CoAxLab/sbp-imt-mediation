#!/bin/bash

m="sbp_auc_g_both"
y="mavg_bulbf_ccaf"
task="both"

case_dir="Y-${y}_M-${m}_task-${task}"

#for model in ridge lasso elasticnet
for model in ridge
do
	output_dir="results/${case_dir}/phenotypes/${model}"
	cmd="python 03-phenotypes.py --target ${y} --mediator ${m} --task ${task} --model ${model} --output_dir ${output_dir}"
	eval $cmd
done

