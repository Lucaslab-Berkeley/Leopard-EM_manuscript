#!/bin/bash


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
PROJECT_DIR="/data/papers/Leopard-EM_paper_data/xe30kv"
MICROGRAPHS_DIR="${PROJECT_DIR}/all_mgraphs"
OUTPUT_DIR="${PROJECT_DIR}/results_constrained_step1_noB"
LARGE_RESULTS_DIR="${PROJECT_DIR}/results_refine_tm_60S_noB"
SMALL_RESULTS_DIR="${PROJECT_DIR}/results_match_tm_40S-body_noB"
TEMPLATE_YAML="${PROJECT_DIR}/configs/constrained_search_config_SSU-body_step1.yaml"
TEMPLATE_DIR="/data/papers/Leopard-EM_paper_data/maps2/SSU-body_map_px0.936_bscale0.5.mrc"
LARGE_SUFFIX="_refined_results.csv"
SMALL_SUFFIX="_results.csv"
SCRIPT_PATH="${PROJECT_DIR}/process_all_micrographs_constrained.py"

# Create results directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Run the processing script
python ${SCRIPT_PATH} \
  --micrographs-dir "${MICROGRAPHS_DIR}" \
  --template-yaml "${TEMPLATE_YAML}" \
  --large-results-dir "${LARGE_RESULTS_DIR}" \
  --small-results-dir "${SMALL_RESULTS_DIR}" \
  --template-volume "${TEMPLATE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --large-suffix "${LARGE_SUFFIX}" \
  --small-suffix "${SMALL_SUFFIX}" \
  --gpus "0,1,2,3" \
  --batch-size 32 \
  --false-positives 0.005

echo "All micrographs processed" 
