#!/usr/bin/env python
import os
import re
import yaml
import sys
import argparse
import glob
from leopard_em.pydantic_models.managers import MatchTemplateManager

def extract_micrograph_number(filename):
    """Extract the micrograph number from the filename."""
    match = re.search(r'xenon_(\d+)_(\d+)_', filename)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return None

def get_ctf_parameters(ctf_file_path):
    """Extract defocus parameters from CTF file."""
    with open(ctf_file_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 5:
                defocus_1 = float(parts[1])  # defocus 1 in Angstroms
                defocus_2 = float(parts[2])  # defocus 2 in Angstroms
                astigmatism_angle = float(parts[3])  # azimuth of astigmatism
                return defocus_1, defocus_2, astigmatism_angle
    return None, None, None

def create_yaml_for_micrograph(template_yaml_path, micrograph_path, defocus_1, defocus_2, astigmatism_angle, output_path, gpu_ids):
    """Create a custom YAML file for a specific micrograph."""
    # Load the template YAML
    with open(template_yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Update the config with the specific micrograph info
    config['micrograph_path'] = micrograph_path
    config['optics_group']['defocus_u'] = defocus_1
    config['optics_group']['defocus_v'] = defocus_2
    config['optics_group']['astigmatism_angle'] = astigmatism_angle
    
    # Set GPU IDs
    config['computational_config']['gpu_ids'] = gpu_ids
    
    # Update output paths to be in the all_results folder with micrograph-specific names
    micrograph_basename = os.path.basename(micrograph_path).split('.')[0]
    results_dir = os.path.dirname(output_path)
    
    for key in config['match_template_result']:
        if key != 'allow_file_overwrite':
            original_path = config['match_template_result'][key]
            filename = os.path.basename(original_path)
            new_filename = f"{micrograph_basename}_{filename}"
            config['match_template_result'][key] = os.path.join(results_dir, new_filename)
    
    # Write the updated config to a new YAML file
    with open(output_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    
    return config

def process_micrograph(micrograph_path, template_yaml, ctfs_dir, output_dir, gpu_ids, batch_size):
    """Process a single micrograph with match template."""
    micrograph_basename = os.path.basename(micrograph_path)
    micrograph_number = extract_micrograph_number(micrograph_basename)
    
    if not micrograph_number:
        print(f"Could not extract micrograph number from {micrograph_basename}")
        return False
    
    # Find corresponding CTF file
    ctf_filename = f"xenon_{micrograph_number}_0.0_diagnostic.txt"
    ctf_file_path = os.path.join(ctfs_dir, ctf_filename)
    
    if not os.path.exists(ctf_file_path):
        print(f"CTF file not found: {ctf_file_path}")
        return False
    
    # Extract defocus parameters
    defocus_1, defocus_2, astigmatism_angle = get_ctf_parameters(ctf_file_path)
    
    if defocus_1 is None:
        print(f"Could not extract defocus parameters from {ctf_file_path}")
        return False
    
    # Create custom YAML file for this micrograph
    custom_yaml_path = os.path.join(output_dir, f"{os.path.splitext(micrograph_basename)[0]}_config.yaml")
    create_yaml_for_micrograph(
        template_yaml, 
        micrograph_path, 
        defocus_1, 
        defocus_2, 
        astigmatism_angle, 
        custom_yaml_path,
        gpu_ids
    )
    
    # Run match template with the custom config
    try:
        print(f"Running match template for {micrograph_basename} with GPUs {gpu_ids}")
        mt_manager = MatchTemplateManager.from_yaml(custom_yaml_path)
        mt_manager.run_match_template(batch_size)
        
        # Save results to CSV
        df = mt_manager.results_to_dataframe()
        csv_path = os.path.join(output_dir, f"{os.path.splitext(micrograph_basename)[0]}_results.csv")
        df.to_csv(csv_path)
        
        print(f"Successfully processed {micrograph_basename}")
        return True
    except Exception as e:
        print(f"Error processing {micrograph_basename}: {str(e)}")
        return False

def is_already_processed(micrograph_path, output_dir):
    """Check if a micrograph has already been processed by looking for its results in output_dir."""
    micrograph_filename = os.path.basename(micrograph_path)
    results_filename = f"{os.path.splitext(micrograph_filename)[0]}_results.csv"
    results_path = os.path.join(output_dir, results_filename)
    return os.path.exists(results_path)

def main():
    parser = argparse.ArgumentParser(description='Process multiple micrographs with match template')
    parser.add_argument('--micrographs-dir', required=True, help='Directory containing micrograph files')
    parser.add_argument('--template-yaml', required=True, help='Path to the template YAML configuration')
    parser.add_argument('--ctfs-dir', required=True, help='Directory containing CTF files')
    parser.add_argument('--output-dir', required=True, help='Directory to store results')
    parser.add_argument('--gpus', default='0', help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--batch-size', type=int, default=8, help='Orientation batch size')
    parser.add_argument('--pattern', default='*DWS.mrc', help='File pattern to match micrographs')
    parser.add_argument('--start-idx', type=int, default=None, help='Start index for processing (optional)')
    parser.add_argument('--end-idx', type=int, default=None, help='End index for processing (optional)')
    parser.add_argument('--job-idx', type=int, default=None, help='Job index from SLURM array (optional)')
    parser.add_argument('--jobs-per-array', type=int, default=None, help='Number of micrographs per array job (optional)')
    
    args = parser.parse_args()
    
    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert GPU IDs string to list of integers
    gpu_ids = [int(gpu_id) for gpu_id in args.gpus.split(',')]
    
    # Get list of micrograph files
    micrograph_pattern = os.path.join(args.micrographs_dir, args.pattern)
    micrograph_files = sorted(glob.glob(micrograph_pattern))
    
    if not micrograph_files:
        print(f"No micrograph files found matching pattern {micrograph_pattern}")
        return 1
    
    print(f"Found {len(micrograph_files)} micrograph files")
    
    # Filter out micrographs that have already been processed
    unprocessed_micrographs = []
    for micrograph_file in micrograph_files:
        if not is_already_processed(micrograph_file, args.output_dir):
            unprocessed_micrographs.append(micrograph_file)
    
    print(f"Found {len(unprocessed_micrographs)} unprocessed micrographs out of {len(micrograph_files)} total")
    
    # Replace the full list with only unprocessed micrographs
    micrograph_files = unprocessed_micrographs
    
    # Check if there are any micrographs to process
    if not micrograph_files:
        print("No unprocessed micrographs to process. Exiting.")
        return 0
    
    # Determine which micrographs to process
    if args.job_idx is not None and args.jobs_per_array is not None:
        # Calculate range for this job in the array
        start_idx = (args.job_idx - 1) * args.jobs_per_array
        end_idx = min(start_idx + args.jobs_per_array, len(micrograph_files))
        micrograph_files = micrograph_files[start_idx:end_idx]
        print(f"Processing micrographs {start_idx+1}-{end_idx} out of {len(micrograph_files)}")
    elif args.start_idx is not None or args.end_idx is not None:
        start_idx = args.start_idx if args.start_idx is not None else 0
        end_idx = args.end_idx if args.end_idx is not None else len(micrograph_files)
        micrograph_files = micrograph_files[start_idx:end_idx]
        print(f"Processing micrographs {start_idx+1}-{end_idx} out of {len(micrograph_files)}")
    
    # Process each micrograph
    successful = 0
    for i, micrograph_file in enumerate(micrograph_files):
        print(f"Processing {i+1}/{len(micrograph_files)}: {os.path.basename(micrograph_file)}")
        if process_micrograph(
            micrograph_file, 
            args.template_yaml, 
            args.ctfs_dir, 
            args.output_dir, 
            gpu_ids, 
            args.batch_size
        ):
            successful += 1
    
    print(f"Successfully processed {successful}/{len(micrograph_files)} micrographs")
    return 0

if __name__ == "__main__":
    sys.exit(main())