import numpy as np
import matplotlib.pyplot as plt
from constrained_thought import gaussian_noise_zscore_cutoff
from matplotlib import rcParams
rcParams['font.family'] = 'Helvetica'


# Parameters
image_size = (4096, 4096)
num_projections = np.logspace(0, np.log10(2E7), 200)  # From 1 to 2E7 projections
pixel_sizes = np.linspace(1, 4096, 200)  # From 1 to 4096 pixels

# Define colors
teal = '#2E8B57'  # Soft teal
coral = '#CD5C5C'  # Muted coral

# Reference point for normalization: 20M projections, 4k×4k image, FPR=1
reference_ccgs = 2E7 * 4096 * 4096
reference_cutoff = gaussian_noise_zscore_cutoff(reference_ccgs, 1)

# Set up figure with specified dimensions (90mm x 60mm)
plt.rcParams['figure.figsize'] = [90/25.4, 60/25.4]  # Convert mm to inches
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({
        'font.size': 7,
        'axes.labelsize': 7,
        'axes.titlesize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'axes.edgecolor': (0, 0, 0, 0.6),  # 60% opacity for axes
        'xtick.color': (0, 0, 0, 0.6),     # 60% opacity for ticks
        'ytick.color': (0, 0, 0, 0.6),     # 60% opacity for ticks
        'axes.labelcolor': (0, 0, 0, 0.6), # 60% opacity for labels
})

# Plot: Varying total number of CCGs (num_projections * num_pixels)
fig, ax = plt.subplots()

# Calculate total CCGs
total_ccgs = np.outer(num_projections, pixel_sizes**2).flatten()
total_ccgs = np.sort(total_ccgs)  # Sort for plotting

# Calculate cutoffs for different total CCGs
cutoffs = []
for n in total_ccgs:
    cutoff = gaussian_noise_zscore_cutoff(n, 1/200)
    cutoffs.append(cutoff)
cutoffs = np.array(cutoffs)

# Calculate relative SNR and molecular mass
relative_snr = cutoffs / reference_cutoff
relative_mass = (cutoffs / reference_cutoff)**2
 
# Plot both metrics
ax.plot(total_ccgs, relative_snr, color=teal, label='Relative SNR', linewidth=1)
ax.plot(total_ccgs, relative_mass, color=coral, label='Relative Mw', linewidth=1)
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Total Cross Correlations')
ax.set_ylabel('Relative Value')
ax.set_xscale('log')
plt.tight_layout()
plt.savefig('total_ccgs_analysis.pdf')

plt.show()

