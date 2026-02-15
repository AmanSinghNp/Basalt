import numpy as np
import time

def benchmark_fk_filter(rows, cols, iterations=10):
    # Generate random data
    data = np.random.rand(rows, cols).astype(np.float32) + \
           1j * np.random.rand(rows, cols).astype(np.float32)
    
    dt = 0.004
    dx = 10.0
    
    # Warmup
    np.fft.fft2(data)
    
    start_time = time.time()
    
    for _ in range(iterations):
        # 1. Forward FFT
        spectrum = np.fft.fft2(data)
        
        # 2. Filter (Simple Mute - emulating likely NumPy implementation)
        # Construct frequency grids
        freqs = np.fft.fftfreq(rows, dt)
        knums = np.fft.fftfreq(cols, dx)
        
        # V = f / k
        # Vectorized mute check
        # This is actually optimistic for Python as it uses optimized C-backend of NumPy
        # but overhead of python loops/allocations will show.
        
        # Meshgrid
        K, F = np.meshgrid(knums, freqs)
        
        # Avoid div zero
        K_safe = np.where(np.abs(K) < 1e-6, 1e-6, K)
        V = F / K_safe
        
        # Mute if |V| < 1500
        mask = np.where(np.abs(V) < 1500, 0.0, 1.0)
        
        filtered = spectrum * mask
        
        # 3. Inverse FFT
        result = np.fft.ifft2(filtered)
        
    end_time = time.time()
    avg_time = (end_time - start_time) / iterations
    return avg_time

if __name__ == "__main__":
    print("Benchmarking Python/NumPy Baseline...")
    sizes = [256, 512, 1024, 2048]
    for size in sizes:
        t = benchmark_fk_filter(size, size)
        print(f"Size {size}x{size}: {t*1000:.2f} ms")
