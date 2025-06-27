import pandas as pd
import mrcfile
import numpy as np
import os
from pathlib import Path
import glob
import starfile

def create_star_file(df, output_dir, output_star_path, box_size) -> None:
    """
    Create a particles.star file from the particle data.
    
    Args:
        df (pandas.DataFrame): DataFrame containing particle data
        output_dir (str): Directory containing the particle stacks
        output_star_path (str): Path to save the star file
        box_size (int): Size of the box used for particle extraction
    """
    # Create a new DataFrame for particles
    particle_df = pd.DataFrame()
    
    # Group by micrograph to get particle indices
    for micrograph_path, particles in df.groupby('micrograph_path'):
        mrcs_name = f"{Path(micrograph_path).stem}.mrcs"
        mrcs_rel_path = os.path.join(output_dir, mrcs_name)
        
        # Add each particle to the particle DataFrame
        for idx, particle in particles.iterrows():
            particle_idx = particles.index.get_loc(idx) + 1  # 1-based index
            particle_data = {
                'rlnCoordinateX': particle['pos_x_img'],
                'rlnCoordinateY': particle['pos_y_img'],
                'rlnImageName': f"{particle_idx:06d}@{mrcs_rel_path}",
                'rlnMicrographName': micrograph_path,  # Use full path
                'rlnOpticsGroup': 1,
                'rlnDefocusU': particle['defocus_u'] + particle['refined_relative_defocus'],
                'rlnDefocusV': particle['defocus_v'] + particle['refined_relative_defocus'],
                'rlnDefocusAngle': particle['astigmatism_angle'],
                'rlnAngleRot': particle['refined_phi'],
                'rlnAngleTilt': particle['refined_theta'],
                'rlnAnglePsi': particle['refined_psi']
            }
            particle_df = pd.concat([particle_df, pd.DataFrame([particle_data])], ignore_index=True)
    
    # Create optics DataFrame
    optics_data = {
        'rlnOpticsGroup': 1,
        'rlnOpticsGroupName': 'OpticsGroup1',
        'rlnMicrographOriginalPixelSize': df['refined_pixel_size'].iloc[0],
        'rlnSphericalAberration': df['spherical_aberration'].iloc[0],
        'rlnVoltage': df['voltage'].iloc[0],
        'rlnAmplitudeContrast': df['amplitude_contrast_ratio'].iloc[0],
        'rlnImagePixelSize': df['refined_pixel_size'].iloc[0],
        'rlnImageSize': box_size
    }
    optics_df = pd.DataFrame([optics_data])
    
    # Write the star file
    starfile.write(
        {'optics': optics_df, 'particles': particle_df},
        output_star_path
    )
    print(f"Created star file: {output_star_path}")

def create_particle_stacks(input_dir, output_dir, box_size, csv_suffix='_results.csv') -> None:
    """
    Create particle stacks from CSV files containing particle positions.
    
    Args:
        input_dir (str): Directory containing input CSV files
        output_dir (str): Directory to save the output stacks
        box_size (int): Size of the square box to crop around each particle
        csv_suffix (str): Suffix pattern to match CSV files (default: '_results.csv')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV files matching the suffix
    csv_pattern = os.path.join(input_dir, f'*{csv_suffix}')
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No CSV files found matching pattern: {csv_pattern}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Create a list to store all particle data
    all_particles = []
    
    # Process each CSV file
    for csv_path in csv_files:
        print(f"\nProcessing CSV file: {csv_path}")
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        all_particles.append(df)
        
        # Group particles by micrograph
        micrograph_groups = df.groupby('micrograph_path')
        
        for micrograph_path, particles in micrograph_groups:
            print(f"Processing micrograph: {micrograph_path}")
            
            # Get pixel size from first particle in the group
            pixel_size = particles.iloc[0]['refined_pixel_size']
            
            # Load the micrograph
            with mrcfile.open(micrograph_path) as mrc:
                micrograph = mrc.data
            
            # Initialize list to store particles
            particle_stack = []
            
            # Process each particle
            for _, particle in particles.iterrows():
                # Get particle center coordinates
                center_x = int(particle['pos_x_img'])
                center_y = int(particle['pos_y_img'])
                
                # Calculate box boundaries
                half_box = box_size // 2
                x_start = max(0, center_x - half_box)
                y_start = max(0, center_y - half_box)
                x_end = min(micrograph.shape[-1], center_x + half_box)
                y_end = min(micrograph.shape[-2], center_y + half_box)
                
                # Skip if the box is empty or invalid
                if x_start >= x_end or y_start >= y_end:
                    print(f"Warning: Invalid box coordinates for particle at ({center_x}, {center_y})")
                    continue
                
                # Extract particle
                particle_img = micrograph[..., y_start:y_end, x_start:x_end]
                
                # Create padded image
                padded_img = np.zeros((box_size, box_size), dtype=micrograph.dtype)
                
                # Calculate the region to copy
                copy_height = min(particle_img.shape[-2], box_size)
                copy_width = min(particle_img.shape[-1], box_size)
                
                # Copy the valid region
                padded_img[:copy_height, :copy_width] = particle_img[..., :copy_height, :copy_width]
                
                particle_stack.append(padded_img)
            
            if not particle_stack:
                print(f"Warning: No valid particles found in {micrograph_path}")
                continue
                
            # Convert list to numpy array
            particle_stack = np.stack(particle_stack)
            
            # Create output filename
            micrograph_name = Path(micrograph_path).stem
            output_path = os.path.join(output_dir, f"{micrograph_name}.mrcs")
            
            # Save the stack
            with mrcfile.new(output_path, overwrite=True) as mrc:
                mrc.set_data(particle_stack)
                mrc.voxel_size = pixel_size  # Use the refined pixel size from the CSV
    
    # Combine all particle data and create star file
    all_particles_df = pd.concat(all_particles, ignore_index=True)
    star_file_path = os.path.join(output_dir, 'particles.star')
    create_star_file(all_particles_df, output_dir, star_file_path, box_size)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create particle stacks from CSV files")
    parser.add_argument("input_dir", help="Directory containing input CSV files")
    parser.add_argument("output_dir", help="Directory to save the output stacks")
    parser.add_argument("--box-size", type=int, default=256, help="Size of the square box to crop around each particle")
    parser.add_argument("--csv-suffix", default='_results.csv',
                      help="Suffix pattern to match CSV files (default: _results.csv)")
    
    args = parser.parse_args()
    
    create_particle_stacks(args.input_dir, args.output_dir, args.box_size, args.csv_suffix) 