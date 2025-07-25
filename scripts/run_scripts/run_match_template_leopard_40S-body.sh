#!/bin/bash
#SBATCH --job-name=match_template
#SBATCH --account=pc_lucaslab
#SBATCH --partition=es1
#SBATCH --qos=es_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/global/scratch/users/jdickerson/2dtm_test_data/xe30kv/logs/match_template_40S_%A_%a.out
#SBATCH --error=/global/scratch/users/jdickerson/2dtm_test_data/xe30kv/logs/match_template_40S_%A_%a.err
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:8

mkdir -p /global/scratch/users/jdickerson/2dtm_test_data/xe30kv/logs
# Load any necessary modules (adjust for your system)
# Print current shell and environment before activation
echo "=== ENVIRONMENT BEFORE ACTIVATION ==="
echo "Current shell: $SHELL"
echo "Current conda environments:"
conda env list
echo "Current Python: $(which python)"
echo "Current Python version: $(python --version 2>&1)"
echo "======================================"

# Activate leopard-em conda environment 
echo "=== ACTIVATING CONDA ENVIRONMENT ==="
source $(conda info --base)/etc/profile.d/conda.sh
conda activate leopard-em
ACTIVATION_STATUS=$?

# Check if activation succeeded
if [ $ACTIVATION_STATUS -ne 0 ]; then
    echo "ERROR: Failed to activate the leopard-em environment"
    echo "Available environments:"
    conda env list
    exit 1
fi

# Print environment details after activation
echo "=== ENVIRONMENT AFTER ACTIVATION ==="
echo "Active conda environment: $CONDA_PREFIX"
echo "Python interpreter: $(which python)"
echo "Python version: $(python --version 2>&1)"
echo "Conda packages in environment:"
conda list | grep -E 'program|leopard'
echo "======================================"


# Set up paths (update these paths for your project)
PROJECT_DIR="/global/scratch/users/jdickerson/2dtm_test_data/xe30kv"
MICROGRAPHS_DIR="${PROJECT_DIR}/all_mgraphs"
OUTPUT_DIR="${PROJECT_DIR}/results_match_tm_40S-body_noB"
CTFS_DIR="${PROJECT_DIR}/all_ctfs"
TEMPLATE_YAML="${PROJECT_DIR}/match_template_config_40S_base.yaml"
SCRIPT_PATH="${PROJECT_DIR}/process_all_micrographs.py"

# Create results directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Run the processing script
python ${SCRIPT_PATH} \
  --micrographs-dir "${MICROGRAPHS_DIR}" \
  --template-yaml "${TEMPLATE_YAML}" \
  --ctfs-dir "${CTFS_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --gpus "0,1,2,3,4,5,6,7" \
  --batch-size 8

echo "All micrographs processed"
