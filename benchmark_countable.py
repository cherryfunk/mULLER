import subprocess
import time
import math
import os

def get_precision(val, target=0.015625):
    error = abs(val - target)
    if error == 0:
        error = 1e-16
    elif math.isnan(error):
        return 0
    return max(0, min(16, -math.log10(error)))

print("====================================")
print("Benchmarking Countable Interpretation")
print("====================================")

# 1. Build
print("Building project...")
subprocess.run(["cabal", "build", "daniel-haskell"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
bin_path = subprocess.check_output(["cabal", "list-bin", "daniel-haskell"]).decode("utf-8").strip()

# 2. Baseline
print("Calculating OS overhead baseline...")
baseline_time = 0
for _ in range(5):
    s = time.perf_counter()
    subprocess.run([bin_path, "baseline"], stdout=subprocess.DEVNULL)
    baseline_time += (time.perf_counter() - s)
baseline_time /= 5

# 3. Execution
scenarios = [
    {"name": "Simplify\n(Algebraic)", "cmd": "benchmark-countable", "target": 0.015625, "color": "green"},
    {"name": "Lazy Eval\n(Bounded)", "cmd": "benchmark-countable-lazy", "target": 0.875, "color": "blue"},
    {"name": "Monte Carlo\n(Heavy Tail)", "cmd": "benchmark-countable-heavy", "target": 1.0, "color": "orange"}
]

print(f"Executing benchmarks...")
results = []

for s in scenarios:
    avg_time = 0
    runs = 10
    val = 0.0
    for _ in range(runs):
        start = time.perf_counter()
        out = subprocess.check_output([bin_path, s["cmd"]]).decode("utf-8")
        end = time.perf_counter()
        avg_time += (end - start)
        val = float(out.strip())

    avg_time /= runs
    pure_time = max(0.001, (avg_time - baseline_time) * 1000)
    digits = get_precision(val, target=s["target"])
    
    results.append({**s, "digits": digits, "time": pure_time})

    clean_name = s['name'].replace('\n',' ')
    print(f"\n--- {clean_name} ---")
    print(f"Evaluated Value: {val:.8f}")
    print(f"Exact Target:    {s['target']:.8f}")
    print(f"Precision:       ~{digits:.1f} digits")
    print(f"Pure Compute:    {pure_time:.4f} ms")
    print("----------------")

# 4. Plotting
try:
    import matplotlib.pyplot as plt
    
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    
    names = [r["name"] for r in results]
    precisions = [r["digits"] for r in results]
    colors = [r["color"] for r in results]
    
    bars = plt.bar(names, precisions, color=colors, alpha=0.7, edgecolor='black', linewidth=1.2)
    
    plt.title("Integration Strategy Precision (Countably Infinite Sets)", fontsize=14, fontweight='bold')
    plt.ylabel("Precision (Correct Decimal Places)", fontsize=12)
    plt.ylim(0, 18)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.3, f"{yval:.1f} digits", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.abspath("countable_benchmark.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\nSaved beautiful countable benchmark plot to: {plot_path}")

except ImportError:
    print("\nMatplotlib not found, skipping plot generation.")
