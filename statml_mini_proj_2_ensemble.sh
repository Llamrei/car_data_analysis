#!/bin/bash
#PBS -o /rds/general/user/al3615/home/car_data_analysis/pbs_junk
#PBS -e /rds/general/user/al3615/home/car_data_analysis/pbs_junk
# Debug queue:
# PBS -l select=1:ncpus=8:mem=32gb:ompthreads=8
# PBS -l walltime=00:29:59
# PBS -J 1-6

# Throughput queue:
#PBS -l select=1:ncpus=8:mem=24gb:ompthreads=8
#PBS -l walltime=23:00:0
#PBS -J 1-181

# GPU queue:
# PBS -lselect=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=RTX6000
# PBS -l walltime=10:00:0
# PBS -J 1-16

echo "Started"
echo "StatML Mini-project 2 - al3615"
date
set -x
job="${PBS_JOBID%.*}"
job="${job%[*}"
working_dir=$HOME/car_data_analysis

# exp_vals will give the name of the scripts to run with the correct models
exp_vals=($(ls -d $working_dir/scripts/*))

echo "Array idx $PBS_ARRAY_INDEX"
if [[ ! -z "$PBS_ARRAY_INDEX" ]]; then
    no_exps="${#exp_vals[@]}"
    array_idx=$PBS_ARRAY_INDEX

    # For repeating experiments we want to cycle through our experiment values
    exp_idx=$(($array_idx % $no_exps))
    exp_val=${exp_vals[$exp_idx]}
else
    exp_val=${exp_vals[0]}
    echo "Not in array job, exp val $exp_val"
fi
# exp_val="$working_dir/scripts/NLP.py"
# array_idx=0

extension=${exp_val#*.}
method_name=`basename ${exp_val%.*}`
progress_file="${method_name}_progress.txt"

if [[ $extension == "r" ]]; then
    script_handler=Rscript
elif [[ $extension == "py" ]]; then
    script_handler=python
fi

exp_identifier=$(($array_idx / 6))

input_data_dir="$working_dir/final_project_data"
output_dir="$working_dir/results/ensemble/$exp_identifier"
progress_file="$output_dir/$progress_file"

mkdir -p $output_dir

module load anaconda3/personal
module load cuda
# Switch on correct conda environment
source activate tf2-text-and-r

# To fix issues with downloading BERT models
export TFHUB_CACHE_DIR=$EPHEMERAL/.cache/tfhub_modules
mkdir -p $TFHUB_CACHE_DIR

touch $progress_file
$script_handler $exp_val $job $array_idx $input_data_dir $output_dir

# Update counter in progress file for how many tests we have
# use flock for atomic transaction
( flock -x 200;
echo $(expr $(cat "$progress_file") + 1) > $progress_file;
) 200>/rds/general/user/al3615/home/car_data_analysis/lockfile 

echo "Done"
date
