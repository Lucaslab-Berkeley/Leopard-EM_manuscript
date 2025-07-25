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


# Activate leopard-em conda environment


# Set up paths (update these paths for your project)
PROJECT_DIR="/data/papers/Leopard-EM_paper_data/xe30kv"
MICROGRAPHS_DIR="${PROJECT_DIR}/all_mgraphs"
OUTPUT_DIR="${PROJECT_DIR}/results_refine_tm_60S_noB"
MATCH_RESULTS_DIR="${PROJECT_DIR}/results_match_tm_60S_noB"
TEMPLATE_YAML="${PROJECT_DIR}/refine_template_config_60S.yaml"
TEMPLATE_DIR="/data/papers/Leopard-EM_paper_data/maps2/60S_map_px0.936_bscale0.5.mrc"
RESULTS_SUFFIX="_results.csv"
SCRIPT_PATH="${PROJECT_DIR}/process_all_micrographs_refine.py"

# Create results directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Run the processing script
python ${SCRIPT_PATH} \
  --micrographs-dir "${MICROGRAPHS_DIR}" \
  --template-yaml "${TEMPLATE_YAML}" \
  --match-results-dir "${MATCH_RESULTS_DIR}" \
  --template-volume "${TEMPLATE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --results-suffix "${RESULTS_SUFFIX}" \
  --gpus "0,1,2,3" \
  --batch-size 64 \
  --pattern "*.mrc"

echo "All micrographs processed"
