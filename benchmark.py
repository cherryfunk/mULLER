import os
import re
import subprocess
import time
import math
import sys

try:
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "matplotlib", "--user"])
    import matplotlib.pyplot as plt

GIRY_PATH = "daniel_haskell/NeSyFramework/Monads/Giry.hs"
NESY_PATH = "Haskell/NeSy.hs"

giry_configs = [
    {"name": "Ultra-Fast", "iter": 50, "prec": "1e-3", "sigma": 3.0},
    {"name": "ML Default", "iter": 100, "prec": "1e-6", "sigma": 4.0},
    {"name": "High", "iter": 500, "prec": "1e-9", "sigma": 6.0},
    {"name": "Extreme", "iter": 1000, "prec": "1e-12", "sigma": 8.0}
]

nesy_configs = [
    {"name": "1k Samples", "samples": 1000},
    {"name": "10k Samples", "samples": 10000},
    {"name": "100k Samples", "samples": 100000},
    {"name": "1M Samples", "samples": 1000000},
    {"name": "10M Samples", "samples": 10000000}
]

nesy_integrator_configs = [
    {"name": "monad-bayes (Integrator)", "samples": 0}
]

with open(GIRY_PATH, "r") as f:
    original_giry = f.read()

with open(NESY_PATH, "r") as f:
    original_nesy = f.read()

def inject_giry_config(cfg):
    content = original_giry
    content = re.sub(r"giryQuadMaxIter = \d+", f"giryQuadMaxIter = {cfg['iter']}", content)
    content = re.sub(r"giryQuadPrecision = [\d.e-]+", f"giryQuadPrecision = {cfg['prec']}", content)
    content = re.sub(r"giryTailSigmas = [\d.e-]+", f"giryTailSigmas = {cfg['sigma']}", content)
    with open(GIRY_PATH, "w") as f:
        f.write(content)

def inject_nesy_config(cfg):
    content = original_nesy
    content = re.sub(r"no_samples2 = \d+", f"no_samples2 = {cfg['samples']}", content)
    with open(NESY_PATH, "w") as f:
        f.write(content)

giry_results = []
nesy_results = []
nesy_int_results = []

def get_precision(val):
    error = abs(val - 0.25)
    if error == 0:
        error = 1e-16
    elif math.isnan(error):
        return 0
    return max(0, min(16, -math.log10(error)))

print("====================================")
print("1. Benchmarking Giry (New Framework)")
print("====================================")

for cfg in giry_configs:
    inject_giry_config(cfg)
    subprocess.run(["cabal", "build", "daniel-haskell"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    bin_path = subprocess.check_output(["cabal", "list-bin", "daniel-haskell"]).decode("utf-8").strip()
    
    # Calculate baseline OS startup overhead time
    baseline_time = 0
    for _ in range(5):
        s = time.perf_counter()
        subprocess.run([bin_path, "baseline"], stdout=subprocess.DEVNULL)
        baseline_time += (time.perf_counter() - s)
    baseline_time /= 5
    
    # Measure execution
    avg_total_time = 0
    runs = 5
    weather_val = float('nan')
    for i in range(runs):
        start = time.perf_counter()
        out = subprocess.check_output([bin_path, "benchmark"]).decode("utf-8")
        end = time.perf_counter()
        avg_total_time += (end - start)
        lines = out.strip().split("\n")
        try:
            weather_val = float(lines[-1].strip())
        except:
            pass
            
    avg_total_time /= runs
    pure_compute_time = max(0.1, (avg_total_time - baseline_time) * 1000) # strictly pure compute time in MS, min 0.1ms 
    
    digits = get_precision(weather_val)
    giry_results.append({
        "name": cfg["name"],
        "time": pure_compute_time,
        "digits": digits,
        "val": weather_val
    })
    print(f"[{cfg['name']}] -> Pure Compute: {pure_compute_time:.2f}ms | OS Overhead: {baseline_time*1000:.2f}ms | Dgt: ~{digits:.1f}")

print("\n====================================")
print("2. Benchmarking NeSy (Monte Carlo)")
print("====================================")

for cfg in nesy_configs:
    inject_nesy_config(cfg)
    subprocess.run(["cabal", "build", "muller"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    bin_path = subprocess.check_output(["cabal", "list-bin", "muller"]).decode("utf-8").strip()
    
    # Simple arbitrary baseline subtraction for MC just to keep scaling fair (since MC has similar OS overhead)
    # We use roughly the same overhead as Giry
    avg_time = 0
    runs = 1
    weather_val = float('nan')
    for i in range(runs):
        start = time.perf_counter()
        out = subprocess.check_output([bin_path]).decode("utf-8")
        end = time.perf_counter()
        avg_time += (end - start)
        
        lines = out.strip().split("\n")
        idx = -1
        for j, line in enumerate(lines):
            if "frequency of True" in line:
                idx = j + 1
                break
        if idx != -1 and idx < len(lines):
            try:
                weather_val = float(lines[idx].strip())
            except:
                pass
            
    avg_time /= runs
    pure_compute_time = max(0.1, (avg_time - baseline_time) * 1000)
    digits = get_precision(weather_val)
    
    nesy_results.append({
        "name": cfg["name"],
        "time": pure_compute_time,
        "digits": digits,
        "val": weather_val
    })
    print(f"[{cfg['name']}] -> Pure Compute: {pure_compute_time:.2f}ms | Val: {weather_val} | Dgt: ~{digits:.1f}")

print("\n====================================")
print("3. Benchmarking NeSy (Native Integrator)")
print("====================================")

for cfg in nesy_integrator_configs:
    # No need to inject since integrator doesn't use samples, just run
    subprocess.run(["cabal", "build", "muller"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    bin_path = subprocess.check_output(["cabal", "list-bin", "muller"]).decode("utf-8").strip()
    
    avg_time = 0
    runs = 3
    weather_val = float('nan')
    for i in range(runs):
        start = time.perf_counter()
        out = subprocess.check_output([bin_path, "integrator"]).decode("utf-8")
        end = time.perf_counter()
        avg_time += (end - start)
        
        lines = out.strip().split("\n")
        idx = -1
        for j, line in enumerate(lines):
            if "frequency of True" in line:
                idx = j + 1
                break
        if idx != -1 and idx < len(lines):
            try:
                weather_val = float(lines[idx].strip())
            except:
                pass
            
    avg_time /= runs
    pure_compute_time = max(0.1, (avg_time - baseline_time) * 1000)
    digits = get_precision(weather_val)
    
    nesy_int_results.append({
        "name": cfg["name"],
        "time": pure_compute_time,
        "digits": digits,
        "val": weather_val
    })
    print(f"[{cfg['name']}] -> Pure Compute: {pure_compute_time:.2f}ms | Val: {weather_val} | Dgt: ~{digits:.1f}")

# Restore
with open(GIRY_PATH, "w") as f: f.write(original_giry)
with open(NESY_PATH, "w") as f: f.write(original_nesy)

# Plotting
plt.style.use('ggplot')
plt.figure(figsize=(15, 8)) # Made chart much wider and taller

# Plot Giry
g_times = [r["time"] for r in giry_results]
g_digits = [r["digits"] for r in giry_results]
g_names = [r["name"] for r in giry_results]
plt.plot(g_times, g_digits, marker='o', markersize=10, linestyle='-', linewidth=2, color='b', label='New (Analytical Quadrature)')

for i, name in enumerate(g_names):
    plt.annotate(name, (g_times[i], g_digits[i]), xytext=(12, 0), textcoords='offset points', fontsize=11, color='b', horizontalalignment='left', verticalalignment='center')

# Plot NeSy
n_times = [r["time"] for r in nesy_results]
n_digits = [r["digits"] for r in nesy_results]
n_names = [r["name"] for r in nesy_results]
plt.plot(n_times, n_digits, marker='^', markersize=10, linestyle='--', linewidth=2, color='red', label='Old (Monte Carlo)')

for i, name in enumerate(n_names):
    if i == 3: # 1M Samples
        plt.annotate(name, (n_times[i], n_digits[i]), xytext=(-5, 25), textcoords='offset points', fontsize=11, color='darkred', horizontalalignment='left', verticalalignment='center')
    elif i < 3: # 1k, 10k, 100k
        off_y = [-22, -10, 0][i]
        plt.annotate(name, (n_times[i], n_digits[i]), xytext=(-10, off_y), textcoords='offset points', fontsize=11, color='darkred', horizontalalignment='right', verticalalignment='center')
    else: # 10M
        plt.annotate(name, (n_times[i], n_digits[i]), xytext=(12, 22), textcoords='offset points', fontsize=11, color='darkred', horizontalalignment='left', verticalalignment='center')

# Plot NeSy Integrator
ni_times = [r["time"] for r in nesy_int_results]
ni_digits = [r["digits"] for r in nesy_int_results]
ni_names = [r["name"] for r in nesy_int_results]
plt.plot(ni_times, ni_digits, marker='*', markersize=15, linestyle='none', color='purple', label='Old (monad-bayes Integrator)')

for i, name in enumerate(ni_names):
    plt.annotate(name, (ni_times[i], ni_digits[i]), xytext=(12, 0), textcoords='offset points', fontsize=11, color='purple', horizontalalignment='left', verticalalignment='center')

plt.title("Pure Compute Runtime vs. Precision: New vs. Old", fontsize=14, fontweight='bold')
plt.xlabel("Pure Mathematical Compute Time (milliseconds) - Log Scale", fontsize=12)
plt.ylabel("Precision (Correct Decimal Places)", fontsize=12)
plt.xscale('log')
plt.grid(True, which="both", linestyle='--', alpha=0.7)
plt.legend(loc='upper right', fontsize=12)

# Set axes limits
plt.ylim(0, 16.5)
all_times = g_times + n_times + ni_times
if len(all_times) > 0:
    plt.xlim(min(all_times) * 0.5, max(all_times) * 10.0) # More buffer for 10M samples and Native Int

plot_path = os.path.abspath("pure_comparison_benchmark.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nSaved beautiful pure comparison plot to: {plot_path}")
