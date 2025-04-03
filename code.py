"""
Foredune Evolution Model
Author: [Your Name]
Date: [Current Date]

This code implements the foredune evolution model described in:
Davidson-Arnott et al. (2018) "Sediment budget controls on foredune height: Comparing simulation model results with field data"
Earth Surface Processes and Landforms, 43: 1798-1810.

The mathematical basis is from the accompanying document "Mathematics from Davidson-Arnott et al., 2018" by Peter Lelievre.

The code simulates foredune growth over 100 years with:
- Annual sediment input of 5 m²/year
- Erosional events determined by random numbers from R1.txt
- Fixed stoss slope angle of 30 degrees
- Fixed lee slope angle of 20 degrees
- Initial dune height of 3 m

The output recreates Figure 7B from the article showing decadal changes in foredune height and width.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Constants from the mathematical summary
Z_INITIAL = 3.0  # Initial dune height (m)
THETA1 = np.radians(30)  # Stoss slope angle (radians)
THETA2 = np.radians(20)  # Lee slope angle (radians)
ANNUAL_SEDIMENT_SUPPLY = 5.0  # Annual sediment supply (m²/year)

# Potential erosion event parameters
EROSION_WIDTHS = np.array([2.5, 5.0, 7.5])  # Potential erosion widths (m)
EROSION_PROBS = np.array([0.09, 0.03, 0.01])  # Probabilities for each erosion width
NON_EROSION_PROB = 0.87  # Probability of no erosion event

# Calculate initial dune dimensions (year 0)
def calculate_initial_dimensions():
    """
    Calculate initial dimensions of the triangular dune using Approach 2 from the math summary.
    
    Returns:
        tuple: (x1, x2, x, z, a1, a2, a) - initial dimensions and areas
    """
    z = Z_INITIAL
    x1 = z / np.tan(THETA1)  # Stoss side width
    x2 = z / np.tan(THETA2)  # Lee side width
    x = x1 + x2  # Total dune width
    
    a1 = 0.5 * x1 * z  # Stoss side area
    a2 = 0.5 * x2 * z  # Lee side area
    a = a1 + a2  # Total dune area
    
    return x1, x2, x, z, a1, a2, a

# Calculate potential erosion areas for each event width
def calculate_potential_erosion_areas():
    """
    Calculate potential erosion areas for each event width using equation (5) from math summary.
    
    Returns:
        np.array: Potential erosion areas corresponding to EROSION_WIDTHS
    """
    return 0.5 * EROSION_WIDTHS**2 * np.tan(THETA1)

# Calculate new dune dimensions after deposition
def update_dune_dimensions(a, a_initial, x_initial, z_initial):
    """
    Update dune dimensions after deposition using Approach 2 (geometric ratios) from math summary.
    
    Args:
        a (float): Current total dune area
        a_initial (float): Initial total dune area
        x_initial (float): Initial total dune width
        z_initial (float): Initial dune height
    
    Returns:
        tuple: (x, z) - new width and height
    """
    f = np.sqrt(a / a_initial)
    x = f * x_initial
    z = f * z_initial
    return x, z

# Calculate erosional wedge dimensions
def calculate_erosion_wedge(a_w, a1_initial, x1_initial):
    """
    Calculate erosional wedge width using equation (7) from math summary.
    
    Args:
        a_w (float): Current erosional wedge area
        a1_initial (float): Initial stoss side area
        x1_initial (float): Initial stoss side width
    
    Returns:
        float: Width of erosional wedge
    """
    return x1_initial * np.sqrt(a_w / a1_initial)

# Main simulation function
def simulate_foredune_evolution(random_numbers, years=100):
    """
    Simulate foredune evolution over specified years.
    
    Args:
        random_numbers (list): List of random numbers for determining erosion events
        years (int): Number of years to simulate
    
    Returns:
        dict: Dictionary containing annual dune dimensions and erosion events
    """
    # Initialize dune dimensions
    x1, x2, x, z, a1, a2, a = calculate_initial_dimensions()
    x_initial = x
    z_initial = z
    a_initial = a
    a1_initial = a1
    
    # Calculate potential erosion areas
    a_p_values = calculate_potential_erosion_areas()
    
    # Initialize variables to track evolution
    a_w = 0  # Erosional wedge area
    x_w = 0  # Erosional wedge width
    
    # Store annual results
    results = {
        'year': [],
        'x': [],
        'z': [],
        'a': [],
        'a_w': [],
        'x_w': [],
        'erosion_event': []
    }
    
    # Add initial conditions
    results['year'].append(0)
    results['x'].append(x)
    results['z'].append(z)
    results['a'].append(a)
    results['a_w'].append(a_w)
    results['x_w'].append(x_w)
    results['erosion_event'].append(0)
    
    # Run simulation for each year
    for year in range(1, years + 1):
        # Determine erosion event type based on random number
        r = random_numbers[year - 1]  # Using 0-based index
        
        if r < NON_EROSION_PROB:
            # No erosion event (87% probability)
            event_type = 0
            a_e = 0
        elif r < NON_EROSION_PROB + EROSION_PROBS[0]:
            # Small erosion event (2.5m, 9% probability)
            event_type = 1
            a_p = a_p_values[0]
            a_e = max(0, a_p - a_w)  # Actual erosion can't exceed potential minus existing wedge
        elif r < NON_EROSION_PROB + EROSION_PROBS[0] + EROSION_PROBS[1]:
            # Medium erosion event (5.0m, 3% probability)
            event_type = 2
            a_p = a_p_values[1]
            a_e = max(0, a_p - a_w)
        else:
            # Large erosion event (7.5m, 1% probability)
            event_type = 3
            a_p = a_p_values[2]
            a_e = max(0, a_p - a_w)
        
        # Update erosional wedge
        a_w += a_e
        
        # Calculate erosional wedge width
        if a_w > 0:
            x_w = calculate_erosion_wedge(a_w, a1_initial, x1_initial)
        else:
            x_w = 0
        
        # Handle sediment deposition
        if a_w > 0:
            # First fill erosional wedge
            a_d = max(0, ANNUAL_SEDIMENT_SUPPLY - a_w)
            a_w = max(0, a_w - ANNUAL_SEDIMENT_SUPPLY)
        else:
            # All sediment goes to dune growth
            a_d = ANNUAL_SEDIMENT_SUPPLY
        
        # Update dune area if there's deposition
        if a_d > 0:
            a += a_d
            x, z = update_dune_dimensions(a, a_initial, x_initial, z_initial)
        
        # Store results for this year
        results['year'].append(year)
        results['x'].append(x)
        results['z'].append(z)
        results['a'].append(a)
        results['a_w'].append(a_w)
        results['x_w'].append(x_w)
        results['erosion_event'].append(event_type)
    
    return results

# Function to plot the dune profile similar to Figure 7B
def plot_dune_profiles(results, decade_years):
    """
    Plot dune profiles at specified decades similar to Figure 7B in the article.
    
    Args:
        results (dict): Simulation results from simulate_foredune_evolution()
        decade_years (list): List of years to plot (e.g., [10, 20, ..., 100])
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors for different decades
    colors = plt.cm.viridis(np.linspace(0, 1, len(decade_years)))
    
    for i, year in enumerate(decade_years):
        idx = year  # Since year 0 is index 0
        z = results['z'][idx]
        x_total = results['x'][idx]
        x_w = results['x_w'][idx]
        
        # Calculate stoss and lee side widths (ignoring erosional wedge for simplicity)
        x1 = z / np.tan(THETA1)  # Stoss side width
        x2 = z / np.tan(THETA2)  # Lee side width
        
        # Create polygon points for the dune profile
        # Starting from toe (0,0)
        points = [(0, 0)]
        
        # Add stoss slope points (with erosional wedge if present)
        if x_w > 0:
            # Erosional wedge present - vertical scarp
            points.append((x_w, 0))
            points.append((x_w, z - (x1 - x_w) * np.tan(THETA1)))
        else:
            # No erosional wedge - normal stoss slope
            points.append((x1, z))
        
        # Add crest and lee slope
        points.append((x1 + x2, 0))
        
        # Create polygon
        poly = Polygon(points, closed=False, fill=False, 
                      edgecolor=colors[i], linewidth=2,
                      label=f'Year {year}')
        ax.add_patch(poly)
    
    # Set plot properties
    ax.set_xlabel('Horizontal Distance (m)')
    ax.set_ylabel('Height (m)')
    ax.set_title('Foredune Evolution with Erosional Events (Figure 7B)')
    ax.legend()
    ax.grid(True)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 20)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('foredune_evolution_figure7b.png')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load random numbers from file (provided by professor)
    try:
        with open('R1.txt', 'r') as f:
            random_numbers = [float(line.strip()) for line in f if line.strip()]
    except FileNotFoundError:
        print("Warning: R1.txt not found. Using generated random numbers instead.")
        np.random.seed(42)  # For reproducibility
        random_numbers = np.random.rand(100).tolist()  # 100 years
    
    # Run simulation
    results = simulate_foredune_evolution(random_numbers, years=100)
    
    # Plot results for decades (10, 20, ..., 100)
    plot_dune_profiles(results, decade_years=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    
    # Print some summary information
    print("Simulation completed successfully.")
    print(f"Final dune height: {results['z'][-1]:.2f} m")
    print(f"Final dune width: {results['x'][-1]:.2f} m")
