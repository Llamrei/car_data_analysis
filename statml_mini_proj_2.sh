#!/bin/bash
# Debug queue:
#PBS -l select=1:ncpus=8:mem=32gb:ompthreads=8
#PBS -l walltime=00:29:59
# PBS -J 1-16

# Throughput queue:
# PBS -l select=1:ncpus=8:mem=24gb:ompthreads=8
# PBS -l walltime=02:00:0
# PBS -J 1-16

# GPU queue:
# PBS -lselect=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=RTX6000
# PBS -l walltime=10:00:0
# PBS -J 1-16

echo "Started"
echo "StatML Mini-project 2 - al3615"
date
set -x
job="${PBS_JOBID%[*}"
working_dir=$HOME/car_data_analysis

# exp_vals will give the name of the scripts to run with the correct models
exp_vals=($(ls -d $working_dir/scripts/*))
extension=${exp_val#*.}

if [[ $extension == "r" ]]; then
    script_handler=Rscript
elif [[ $extension == "py" ]]; then
    script_handler=python
fi

echo "Array idx $PBS_ARRAY_INDEX"
if [[ ! -z "$PBS_ARRAY_INDEX" ]]; then
    no_exps="${#exp_vals[@]}"
    # Why the -1?!?!
    array_idx=$(($PBS_ARRAY_INDEX - 1))

    # For repeating experiments we want to cycle through our experiment values
    exp_idx=$(($array_idx % $no_exps))
    exp_val=${exp_vals[$exp_idx]}
else
    exp_val=${exp_vals[0]}
    echo "Not in array job, exp val $exp_val"
fi

input_data_dir="$working_dir/final_project_data"
output_dir="$working_dir/results"

mkdir -p $output_dir

module load anaconda3/personal
module load cuda
# Switch on correct conda environment
source activate tf2-text-and-r

# To fix issues with downloading BERT models
export TFHUB_CACHE_DIR=$EPHEMERAL/.cache/tfhub_modules
mkdir -p $TFHUB_CACHE_DIR

$script_handler $exp_val $job $array_idx $input_data_dir $output_dir

echo "Done"
date
