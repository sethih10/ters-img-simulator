#!/bin/bash
# TERS data for all of QM9S, as an array of 8 x 32-core jobs on batch-milan.
# 32-core pieces fit into partly used nodes and start quickly; a full 128-core node can wait for days.
# Measured ~1.5 s per molecule per core -> ~130k molecules in ~15 min on 256 cores; the time limit is generous.
#
# Submit from the repository root:
#     sbatch qm9s_adapter/run_qm9s.sbatch
# Resubmitting resumes: molecules with an existing .npz are skipped.
# Other resolution / output folder, e.g. 64x64:
#     OUT=.../qm9s_ters_64 LOGS=.../qm9s_ters_64_logs sbatch qm9s_adapter/run_qm9s.sbatch --grid 64

#SBATCH --job-name=ters-qm9s
#SBATCH --partition=batch-milan
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%A_%a.out

ZIP=/scratch/phys/sin/sethih1/qm9s-raw/qm9s_csv.zip
OUT=${OUT:-/scratch/phys/sin/sethih1/data_files/qm9s_ters}
LOGS=${LOGS:-/scratch/phys/sin/sethih1/data_files/qm9s_ters_logs}

# One process per core: keep numpy from starting its own threads in every worker
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

module load scicomp-python-env

python qm9s_adapter/qm9s_spectrum_generation.py "$ZIP" "$OUT" "$LOGS" \
    --shard "$SLURM_ARRAY_TASK_ID" --num_shards "$SLURM_ARRAY_TASK_COUNT" "$@"
