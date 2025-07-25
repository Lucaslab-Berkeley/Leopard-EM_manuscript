#!/usr/bin/env python
import os
import re
import yaml
import sys
import argparse
import glob
import time
import pandas as pd
from leopard_em.pydantic_models.managers import RefineTemplateManager

def extract_micrograph_number(filename):
    """Extract the micrograph number from the filename."""
    # The new pattern matches filenames like: 25_Sep12_11.40.05_145_1.mrc
    # Simply returns the base filename without extension
    # This makes it work directly with similar result files: 25_Sep12_11.40.05_145_1_results.csv
    base_name = os.path.splitext(filename)[0]
    return base_name

def extract_numeric_micrograph_id(filename):
    """Extract the numeric micrograph ID from filenames like xenon_219_*.mrc"""
    # Look for pattern like xenon_NUMBER_
    match = re.search(r'xenon_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

def create_yaml_for_refinement(template_yaml_path, match_results_csv, output_yaml_path, template_volume_path, gpu_ids):
    """Create a custom YAML file for refinement of a specific micrograph's match results."""
    # Load the template YAML
    with open(template_yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Update the config with the specific match results info
    config['template_volume_path'] = template_volume_path
    config['particle_stack']['df_path'] = match_results_csv
    
    # Set GPU IDs
    config['computational_config']['gpu_ids'] = gpu_ids
    
    # Write the updated config to a new YAML file
    with open(output_yaml_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    
    return config

def process_micrograph_refinement(micrograph_path, match_results_csv, template_yaml, output_dir, template_volume_path, gpu_ids, batch_size):
    """Process refinement for a single micrograph match template results."""
    micrograph_basename = os.path.basename(micrograph_path)
    micrograph_number = extract_micrograph_number(micrograph_basename)
    
    if not micrograph_number:
        print(f"Could not extract micrograph number from {micrograph_basename}")
        return False
    
    # Check if match results CSV exists
    if not os.path.exists(match_results_csv):
        print(f"Match template results not found: {match_results_csv}")
        return False
    
    # Check if there are any results to refine
    df = pd.read_csv(match_results_csv)
    if len(df) == 0:
        print(f"No matches to refine in {match_results_csv}")
        return False
    
    # Create custom YAML file for refinement of this micrograph's results
    custom_yaml_path = os.path.join(output_dir, f"{os.path.splitext(micrograph_basename)[0]}_refine_config.yaml")
    create_yaml_for_refinement(
        template_yaml, 
        match_results_csv, 
        custom_yaml_path,
        template_volume_path,
        gpu_ids
    )
    
    # Run refine template with the custom config
    try:
        print(f"Running refine template for {micrograph_basename} results with GPUs {gpu_ids}")
        rt_manager = RefineTemplateManager.from_yaml(custom_yaml_path)
        
        # Define output path for refinement results
        refine_output_csv = os.path.join(output_dir, f"{os.path.splitext(micrograph_basename)[0]}_refined_results.csv")
        
        # Record start time
        start_time = time.time()
        
        # Run refinement
        rt_manager.run_refine_template(refine_output_csv, batch_size)
        
        # Calculate and print elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print(f"Refinement wall time: {elapsed_time_str}")
        
        print(f"Successfully refined matches for {micrograph_basename}")
        return True
    except Exception as e:
        print(f"Error refining matches for {micrograph_basename}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Process multiple micrographs with refine template')
    parser.add_argument('--micrographs-dir', required=True, help='Directory containing micrograph files')
    parser.add_argument('--template-yaml', required=True, help='Path to the template refine YAML configuration')
    parser.add_argument('--match-results-dir', required=True, help='Directory containing match template results')
    parser.add_argument('--template-volume', required=True, help='Path to the template volume MRC file')
    parser.add_argument('--output-dir', required=True, help='Directory to store refinement results')
    parser.add_argument('--results-suffix', default='_results.csv', help='Suffix for match template results files (default: "_results.csv")')
    parser.add_argument('--gpus', default='0', help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--batch-size', type=int, default=64, help='Particle batch size for refinement')
    parser.add_argument('--pattern', default='*DWS.mrc', help='File pattern to match micrographs')
    parser.add_argument('--start-idx', type=int, default=None, help='Start index for processing (optional)')
    parser.add_argument('--end-idx', type=int, default=None, help='End index for processing (optional)')
    parser.add_argument('--job-idx', type=int, default=None, help='Job index from SLURM array (optional)')
    parser.add_argument('--jobs-per-array', type=int, default=None, help='Number of micrographs per array job (optional)')
    parser.add_argument('--filter-numbers', type=str, default=None, help='Comma-separated list of micrograph numbers to process (e.g., "219,233,249")')
    
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
    
    # Filter by specific micrograph numbers if provided
    if args.filter_numbers:
        filter_numbers = [int(num.strip()) for num in args.filter_numbers.split(',')]
        print(f"Filtering for micrograph numbers: {filter_numbers}")
        
        filtered_files = []
        for mgraph_file in micrograph_files:
            mgraph_id = extract_numeric_micrograph_id(os.path.basename(mgraph_file))
            if mgraph_id in filter_numbers:
                filtered_files.append(mgraph_file)
        
        micrograph_files = filtered_files
        print(f"After filtering: {len(micrograph_files)} micrograph files")
    
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
    
    # Process each micrograph's match results for refinement
    successful = 0
    for i, micrograph_file in enumerate(micrograph_files):
        micrograph_basename = os.path.basename(micrograph_file)
        base_name = os.path.splitext(micrograph_basename)[0]
        
        # Find the corresponding match results CSV file using the configurable suffix
        match_results_csv = os.path.join(args.match_results_dir, f"{base_name}{args.results_suffix}")
        
        print(f"Processing {i+1}/{len(micrograph_files)}: {micrograph_basename}")
        if process_micrograph_refinement(
            micrograph_file, 
            match_results_csv,
            args.template_yaml, 
            args.output_dir,
            args.template_volume,
            gpu_ids, 
            args.batch_size
        ):
            successful += 1
    
    print(f"Successfully refined matches for {successful}/{len(micrograph_files)} micrographs")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 