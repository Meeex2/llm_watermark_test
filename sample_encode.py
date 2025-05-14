import numpy as np
from scipy.stats import norm, truncnorm
from scipy.special import ndtri
import matplotlib.pyplot as plt
import random

class GaussianIntervalSampler:
    """
    Samples values from a Gaussian distribution based on binary code intervals.
    """
    
    @staticmethod
    def binary_to_interval(binary_sequence):
        """
        Convert a binary sequence to its corresponding interval in the normal distribution.
        
        Args:
            binary_sequence: A list or string of 0s and 1s
            
        Returns:
            tuple: (lower_bound, upper_bound) of the interval
        """
        # Convert to list of integers if input is a string
        if isinstance(binary_sequence, str):
            binary_sequence = [int(bit) for bit in binary_sequence]
        
        # Start with the entire probability range [0, 1], with epsilon to avoid inf
        epsilon = 1e-10
        lower_prob = epsilon
        upper_prob = 1.0 - epsilon
        
        # Progressively narrow the probability interval based on each bit
        for bit in binary_sequence:
            mid_prob = (lower_prob + upper_prob) / 2
            
            # Update interval based on the bit
            if bit == 0:
                upper_prob = mid_prob
            else:  # bit == 1
                lower_prob = mid_prob
        
        # Convert probability bounds to normal distribution values
        lower_bound = ndtri(lower_prob)
        upper_bound = ndtri(upper_prob)
        
        return lower_bound, upper_bound
    
    @staticmethod
    def sample_from_interval(lower_bound, upper_bound, size=1):
        """
        Sample from a truncated normal distribution within the specified interval.
        
        Args:
            lower_bound: Lower bound of the interval
            upper_bound: Upper bound of the interval
            size: Number of samples to generate
            
        Returns:
            numpy.ndarray: Samples from the truncated normal distribution
        """
        # Calculate the standardized bounds
        a = (lower_bound - 0) / 1  # (lower - mean) / std
        b = (upper_bound - 0) / 1  # (upper - mean) / std
        
        # Sample from the truncated normal distribution
        samples = truncnorm.rvs(a, b, loc=0, scale=1, size=size)
        
        return samples
    
    @staticmethod
    def sample_from_binary(binary_sequence, size=1):
        """
        Sample from the Gaussian interval corresponding to the binary sequence.
        
        Args:
            binary_sequence: A binary sequence (string or list)
            size: Number of samples to generate
            
        Returns:
            numpy.ndarray: Samples from the corresponding interval
        """
        lower_bound, upper_bound = GaussianIntervalSampler.binary_to_interval(binary_sequence)
        return GaussianIntervalSampler.sample_from_interval(lower_bound, upper_bound, size)


def demonstrate_sampling_with_points():
    """
    Demonstrate the full process of sampling from binary codes using histograms.
    """
    sampler = GaussianIntervalSampler()
    
    # Generate random binary codes and sample from them
    num_samples = 10000
    code_length = 4
    
    # Generate random binary codes
    random_codes = [
        ''.join(str(random.randint(0, 1)) for _ in range(code_length))
        for _ in range(num_samples)
    ]
    
    # Sample from each code's interval
    samples = [sampler.sample_from_binary(code)[0] for code in random_codes]
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    
    # Plot the standard normal distribution for comparison
    x = np.linspace(-4, 4, 1000)
    y = norm.pdf(x)
    plt.plot(x, y, 'r-', lw=2, label='Standard Normal PDF')
    
    # Create a histogram of the samples with small bins
    plt.hist(samples, bins=100, density=True, alpha=0.5, color='skyblue', label='Sample Histogram')
    
    # Add a kernel density estimate to show the empirical distribution
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(samples)
    plt.plot(x, kde(x), 'b-', lw=1.5, label='KDE of Samples')
    
    # Add labels and title
    plt.title('Distribution of Samples from Random Binary Code Intervals', fontsize=14)
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    
    # Print statistics
    mean_sample = np.mean(samples)
    std_sample = np.std(samples)
    print(f"Sample Mean: {mean_sample:.6f}")
    print(f"Sample Std Dev: {std_sample:.6f}")
    print(f"Expected Mean: 0")
    print(f"Expected Std Dev: 1")
    
    plt.tight_layout()
    plt.savefig('histogram_sampling_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_intervals_with_samples():
    """
    Visualize the intervals with sample histograms for each 4-bit sequence.
    """
    sampler = GaussianIntervalSampler()
    
    # Generate all possible 4-bit sequences
    all_sequences = [''.join(map(str, seq)) for seq in np.ndindex((2,2,2,2))]
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the full normal distribution for reference
    x = np.linspace(-4, 4, 1000)
    y = norm.pdf(x)
    ax.plot(x, y, 'k-', lw=2, alpha=0.7, label='Standard Normal PDF')
    
    # Define colors for the intervals
    colors = plt.cm.tab20(np.linspace(0, 1, 16))
    
    # Plot the intervals and samples
    for i, seq in enumerate(sorted(all_sequences)):
        # Get the interval
        lower_bound, upper_bound = sampler.binary_to_interval(seq)
        
        # Generate samples
        samples = sampler.sample_from_binary(seq, size=500)
        
        # Plot the interval as a colored region
        mask = (x >= lower_bound) & (x <= upper_bound)
        ax.fill_between(x[mask], y[mask], alpha=0.3, color=colors[i])
        
        # Add vertical lines at the interval boundaries
        ax.axvline(lower_bound, color=colors[i], linestyle='--', alpha=0.7)
        ax.axvline(upper_bound, color=colors[i], linestyle='--', alpha=0.7)
        
        # Plot the samples as a histogram
        # Clamp bounds to finite values for histogram
        finite_lower = max(lower_bound, -10)  # Reasonable finite bound
        finite_upper = min(upper_bound, 10)
        if finite_lower >= finite_upper:
            continue  # Skip invalid intervals
        try:
            hist, bins = np.histogram(samples, bins=20, range=(finite_lower, finite_upper), density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            # Scale histogram height to be visible but not overpower the PDF
            hist = hist * 0.1 / np.max(hist) if np.max(hist) > 0 else hist
            ax.fill_between(bin_centers, hist + i*0.02, i*0.02, color=colors[i], alpha=0.8, label=seq)
        except ValueError:
            print(f"Skipping histogram for sequence {seq} due to invalid range")
    
    # Add labels and title
    ax.set_title('Intervals and Sample Histograms for All 4-bit Sequences', fontsize=16)
    ax.set_xlabel('Value', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    
    # Add a legend for the sequences
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('intervals_with_sample_histograms.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Demonstrate the sampling method with histograms
    demonstrate_sampling_with_points()
    
    # Visualize intervals with sample histograms
    visualize_intervals_with_samples()
