#!/usr/bin/env python
import os
import re
import yaml
import sys
import argparse
import glob
import time
import pandas as pd
from datetime import datetime
from leopard_em.pydantic_models.managers import ConstrainedSearchManager

# Global error log file handler
error_log_file = None

def log_error(message, also_print=True):
    """Log error message to both console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    
    if also_print:
        print(message)
    
    if error_log_file:
        error_log_file.write(log_message + "\n")
        error_log_file.flush()  # Ensure immediate write

def log_info(message, also_print=True):
    """Log info message to both console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] INFO: {message}"
    
    if also_print:
        print(message)
    
    if error_log_file:
        error_log_file.write(log_message + "\n")
        error_log_file.flush()

def extract_micrograph_number(filename):
    """Extract the micrograph number from the filename."""
    match = re.search(r'xenon_(\d+)_(\d+)_', filename)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return None

def extract_numeric_micrograph_id(filename):
    """Extract the numeric micrograph ID from filenames like xenon_219_*.mrc"""
    # Look for pattern like xenon_NUMBER_
    match = re.search(r'xenon_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

def create_yaml_for_constrained_search(template_yaml_path, large_results_csv, small_results_csv, output_yaml_path, template_volume_path, gpu_ids):
    """Create a custom YAML file for constrained search of a specific micrograph's match results."""
    # Load the template YAML
    with open(template_yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Update the config with the specific match results info
    config['template_volume_path'] = template_volume_path
    config['particle_stack_reference']['df_path'] = large_results_csv
    config['particle_stack_constrained']['df_path'] = small_results_csv
    
    # Set GPU IDs
    config['computational_config']['gpu_ids'] = gpu_ids
    
    # Write the updated config to a new YAML file
    with open(output_yaml_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    
    return config

def process_micrograph_constrained_search(micrograph_path, large_results_csv, small_results_csv, template_yaml, output_dir, template_volume_path, gpu_ids, batch_size, false_positives):
    """Process constrained search for a single micrograph's match template results."""
    micrograph_basename = os.path.basename(micrograph_path)
    micrograph_number = extract_micrograph_number(micrograph_basename)
    
    if not micrograph_number:
        print(f"Could not extract micrograph number from {micrograph_basename}")
        return False
    
    # Check if match results CSV files exist
    if not os.path.exists(large_results_csv):
        print(f"Large particle results not found: {large_results_csv}")
        return False
    
    if not os.path.exists(small_results_csv):
        print(f"Small particle results not found: {small_results_csv}")
        return False
    
    # Check if there are any results to process
    df_large = pd.read_csv(large_results_csv)
    if len(df_large) == 0:
        print(f"No large particle matches in {large_results_csv}")
        return False
    
    #df_small = pd.read_csv(small_results_csv)
    #if len(df_small) == 0:
    #    print(f"No small particle matches in {small_results_csv}")
    #    return False
    
    # Create custom YAML file for constrained search of this micrograph's results
    custom_yaml_path = os.path.join(output_dir, f"{os.path.splitext(micrograph_basename)[0]}_constrained_config.yaml")
    create_yaml_for_constrained_search(
        template_yaml, 
        large_results_csv, 
        small_results_csv,
        custom_yaml_path,
        template_volume_path,
        gpu_ids
    )
    
    # Run constrained search with the custom config
    try:
        log_info(f"Running constrained search for {micrograph_basename} with GPUs {gpu_ids}")
        
        # Pre-flight checks for debugging
        log_info(f"DEBUG: Checking file existence and validity...", also_print=False)
        
        # Check if input files exist
        if not os.path.exists(large_results_csv):
            raise FileNotFoundError(f"Large results CSV not found: {large_results_csv}")
        
        if not os.path.exists(small_results_csv):
            raise FileNotFoundError(f"Small results CSV not found: {small_results_csv}")
        
        if not os.path.exists(template_volume_path):
            raise FileNotFoundError(f"Template volume not found: {template_volume_path}")
        
        if not os.path.exists(custom_yaml_path):
            raise FileNotFoundError(f"Custom YAML config not found: {custom_yaml_path}")
        
        # Check if input CSV files are readable and have content
        try:
            large_df = pd.read_csv(large_results_csv, header=None)
            log_info(f"DEBUG: Large results CSV has {len(large_df)} rows", also_print=False)
            if len(large_df) == 0:
                raise ValueError(f"Large results CSV is empty: {large_results_csv}")
        except Exception as csv_error:
            raise ValueError(f"Error reading large results CSV {large_results_csv}: {str(csv_error)}")
        
        try:
            small_df = pd.read_csv(small_results_csv, header=None)
            log_info(f"DEBUG: Small results CSV has {len(small_df)} rows", also_print=False)
            if len(small_df) == 0:
                raise ValueError(f"Small results CSV is empty: {small_results_csv}")
        except Exception as csv_error:
            raise ValueError(f"Error reading small results CSV {small_results_csv}: {str(csv_error)}")
        
        # Check YAML validity
        try:
            with open(custom_yaml_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
                log_info(f"DEBUG: YAML config loaded successfully", also_print=False)
                log_info(f"DEBUG: YAML keys: {list(yaml_content.keys())}", also_print=False)
        except Exception as yaml_error:
            raise ValueError(f"Error parsing YAML config {custom_yaml_path}: {str(yaml_error)}")
        
        # Check GPU availability (if possible)
        try:
            import torch
            if torch.cuda.is_available():
                available_gpus = torch.cuda.device_count()
                log_info(f"DEBUG: CUDA available, {available_gpus} GPUs detected", also_print=False)
                for gpu_id in gpu_ids:
                    if gpu_id >= available_gpus:
                        log_info(f"WARNING: GPU {gpu_id} not available (only {available_gpus} GPUs detected)")
            else:
                log_info(f"WARNING: CUDA not available, but GPU IDs specified: {gpu_ids}")
        except ImportError:
            log_info(f"DEBUG: torch not available for GPU check", also_print=False)
        
        # Initialize ConstrainedSearchManager
        log_info(f"DEBUG: Creating ConstrainedSearchManager from YAML...", also_print=False)
        cs_manager = ConstrainedSearchManager.from_yaml(custom_yaml_path)
        log_info(f"DEBUG: ConstrainedSearchManager created successfully", also_print=False)
        
        # Define output path for constrained search results
        constrained_output_csv = os.path.join(output_dir, f"{os.path.splitext(micrograph_basename)[0]}_constrained_results.csv")
        log_info(f"DEBUG: Output CSV path: {constrained_output_csv}", also_print=False)
        
        # Check if output directory is writable
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"Output directory not writable: {output_dir}")
        
        # Record start time
        start_time = time.time()
        
        # Run constrained search
        log_info(f"DEBUG: Starting constrained search with batch_size={batch_size}, false_positives={false_positives}", also_print=False)
        cs_manager.run_constrained_search(constrained_output_csv, false_positives, batch_size)
        
        # Calculate and print elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        log_info(f"Constrained search wall time: {elapsed_time_str}")
        
        # Check if output file was created
        if os.path.exists(constrained_output_csv):
            try:
                output_df = pd.read_csv(constrained_output_csv, header=None)
                log_info(f"DEBUG: Output CSV created with {len(output_df)} rows", also_print=False)
            except Exception as output_error:
                log_info(f"WARNING: Output CSV exists but cannot be read: {str(output_error)}")
        else:
            log_info(f"WARNING: Output CSV was not created: {constrained_output_csv}")
        
        log_info(f"Successfully completed constrained search for {micrograph_basename}", also_print=False)
        return True
        
    except FileNotFoundError as e:
        log_error(f"FILE NOT FOUND ERROR for {micrograph_basename}:")
        log_error(f"  {str(e)}")
        log_error(f"  Current working directory: {os.getcwd()}")
        return False
        
    except ValueError as e:
        log_error(f"VALUE ERROR for {micrograph_basename}:")
        log_error(f"  {str(e)}")
        return False
        
    except PermissionError as e:
        log_error(f"PERMISSION ERROR for {micrograph_basename}:")
        log_error(f"  {str(e)}")
        return False
        
    except ImportError as e:
        log_error(f"IMPORT ERROR for {micrograph_basename}:")
        log_error(f"  {str(e)}")
        log_error(f"  This might indicate missing dependencies")
        return False
        
    except Exception as e:
        import traceback
        log_error(f"UNEXPECTED ERROR for {micrograph_basename}:")
        log_error(f"  Error type: {type(e).__name__}")
        log_error(f"  Error message: {str(e)}")
        log_error(f"  Full traceback:")
        
        # Capture traceback to string for logging
        tb_str = traceback.format_exc()
        for line in tb_str.split('\n'):
            if line.strip():
                log_error(f"    {line}")
        
        log_error(f"  Debug information:")
        log_error(f"    - Micrograph path: {micrograph_path}")
        log_error(f"    - Large results CSV: {large_results_csv}")
        log_error(f"    - Small results CSV: {small_results_csv}")
        log_error(f"    - Template YAML: {template_yaml}")
        log_error(f"    - Custom YAML: {custom_yaml_path}")
        log_error(f"    - Template volume: {template_volume_path}")
        log_error(f"    - Output directory: {output_dir}")
        log_error(f"    - GPU IDs: {gpu_ids}")
        log_error(f"    - Batch size: {batch_size}")
        log_error(f"    - False positives: {false_positives}")
        log_error(f"    - Output CSV path: {constrained_output_csv}")
        log_error(f"  " + "="*50)  # Separator line
        return False

def main():
    global error_log_file
    
    parser = argparse.ArgumentParser(description='Process multiple micrographs with constrained search')
    parser.add_argument('--micrographs-dir', required=True, help='Directory containing micrograph files')
    parser.add_argument('--template-yaml', required=True, help='Path to the template constrained search YAML configuration')
    parser.add_argument('--large-results-dir', required=True, help='Directory containing large particle results')
    parser.add_argument('--small-results-dir', required=True, help='Directory containing small particle results')
    parser.add_argument('--template-volume', required=True, help='Path to the template volume MRC file (small particle)')
    parser.add_argument('--output-dir', required=True, help='Directory to store constrained search results')
    parser.add_argument('--large-suffix', default='_refined_results.csv', help='Suffix for large particle result files (default: "_refined_results.csv")')
    parser.add_argument('--small-suffix', default='_results.csv', help='Suffix for small particle result files (default: "_results.csv")')
    parser.add_argument('--gpus', default='0', help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--batch-size', type=int, default=80, help='Particle batch size for constrained search')
    parser.add_argument('--false-positives', type=float, default=0.005, help='False positive rate for constrained search')
    parser.add_argument('--pattern', default='*DWS.mrc', help='File pattern to match micrographs')
    parser.add_argument('--start-idx', type=int, default=None, help='Start index for processing (optional)')
    parser.add_argument('--end-idx', type=int, default=None, help='End index for processing (optional)')
    parser.add_argument('--job-idx', type=int, default=None, help='Job index from SLURM array (optional)')
    parser.add_argument('--jobs-per-array', type=int, default=None, help='Number of micrographs per array job (optional)')
    parser.add_argument('--error-log', default=None, help='Path to error log file (default: constrained_search_errors.log in output directory)')
    parser.add_argument('--filter-numbers', type=str, default=None, help='Comma-separated list of micrograph numbers to process (e.g., "219,233,249")')
    
    args = parser.parse_args()
    
    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize error log file
    if args.error_log:
        error_log_path = args.error_log
    else:
        error_log_path = os.path.join(args.output_dir, 'constrained_search_errors.log')
    
    try:
        error_log_file = open(error_log_path, 'w')
        log_info(f"Starting constrained search processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_info(f"Error log file: {error_log_path}")
        log_info(f"Output directory: {args.output_dir}")
        log_info(f"GPU IDs: {args.gpus}")
        log_info(f"Batch size: {args.batch_size}")
        log_info(f"False positives: {args.false_positives}")
        log_info("="*60)
    except Exception as e:
        print(f"ERROR: Could not create error log file {error_log_path}: {str(e)}")
        return 1
    
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
        log_info(f"Filtering for micrograph numbers: {filter_numbers}")
        
        filtered_files = []
        for mgraph_file in micrograph_files:
            mgraph_id = extract_numeric_micrograph_id(os.path.basename(mgraph_file))
            if mgraph_id in filter_numbers:
                filtered_files.append(mgraph_file)
        
        micrograph_files = filtered_files
        print(f"After filtering: {len(micrograph_files)} micrograph files")
        log_info(f"After filtering: {len(micrograph_files)} micrograph files")
    
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
    
    # Process each micrograph's match results for constrained search
    successful = 0
    try:
        for i, micrograph_file in enumerate(micrograph_files):
            micrograph_basename = os.path.basename(micrograph_file)
            base_name = os.path.splitext(micrograph_basename)[0]
            
            # Find the corresponding results CSV files using the configurable suffixes
            large_results_csv = os.path.join(args.large_results_dir, f"{base_name}{args.large_suffix}")
            small_results_csv = os.path.join(args.small_results_dir, f"{base_name}{args.small_suffix}")
            
            print(f"Processing {i+1}/{len(micrograph_files)}: {micrograph_basename}")
            log_info(f"Processing {i+1}/{len(micrograph_files)}: {micrograph_basename}")
            log_info(f"  Large results CSV: {large_results_csv}")
            log_info(f"  Small results CSV: {small_results_csv}")
            
            if process_micrograph_constrained_search(
                micrograph_file, 
                large_results_csv,
                small_results_csv,
                args.template_yaml, 
                args.output_dir,
                args.template_volume,
                gpu_ids, 
                args.batch_size,
                args.false_positives
            ):
                successful += 1
                log_info(f"SUCCESS: {micrograph_basename} processed successfully")
            else:
                log_info(f"FAILED: {micrograph_basename} failed to process")
            
            log_info("-" * 40)  # Separator between micrographs
    
    except KeyboardInterrupt:
        log_error("Processing interrupted by user (Ctrl+C)")
        print("Processing interrupted by user (Ctrl+C)")
    except Exception as e:
        log_error(f"Unexpected error during processing: {str(e)}")
        print(f"Unexpected error during processing: {str(e)}")
    finally:
        # Ensure log file is closed even if there's an error
        if error_log_file:
            error_log_file.close()
    
    # Log final summary (only if log file is still open)
    if error_log_file and not error_log_file.closed:
        log_info(f"Processing complete!")
        log_info(f"Successfully completed constrained search for {successful}/{len(micrograph_files)} micrographs")
        log_info(f"Failed: {len(micrograph_files) - successful}/{len(micrograph_files)} micrographs")
        log_info(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"Successfully completed constrained search for {successful}/{len(micrograph_files)} micrographs")
    if 'error_log_path' in locals():
        print(f"Detailed error log saved to: {error_log_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 