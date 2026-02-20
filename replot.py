import os
import matplotlib.pyplot as plt

g_times = [0.10, 0.10, 0.10, 0.10]
g_digits = [3.2, 4.8, 9.3, 13.8]
g_names = ["Ultra-Fast", "ML Default", "High", "Extreme"]

n_times = [412.05, 436.29, 532.96, 1504.16, 11303.89]
n_digits = [2.2, 3.2, 4.0, 3.4, 4.5]
n_names = ["1k Samples", "10k Samples", "100k Samples", "1M Samples", "10M Samples"]

ni_times = [929.23]
ni_digits = [2.5]
ni_names = ["Native Integrator"]

plt.style.use('ggplot')
plt.figure(figsize=(15, 8))

# Plot Giry
plt.plot(g_times, g_digits, marker='o', markersize=10, linestyle='-', linewidth=2, color='b', label='New (Analytical Quadrature)')
for i, name in enumerate(g_names):
    plt.annotate(name, (g_times[i], g_digits[i]), xytext=(10, 0), textcoords='offset points', fontsize=11, color='b', horizontalalignment='left', verticalalignment='center')

# Plot NeSy
plt.plot(n_times, n_digits, marker='^', markersize=10, linestyle='--', linewidth=2, color='red', label='Old (Monte Carlo)')

# Labels for red points: 
for i, name in enumerate(n_names):
    if i == 3: # 1M Samples: "JUST a HINT more left and a hint more up"
         plt.annotate(name, (n_times[i], n_digits[i]), xytext=(-5, 25), textcoords='offset points', fontsize=11, color='darkred', horizontalalignment='left', verticalalignment='center')
    elif i < 3: # 1k, 10k, 100k
        off_y = [-22, -10, 0][i]
        plt.annotate(name, (n_times[i], n_digits[i]), xytext=(-10, off_y), textcoords='offset points', fontsize=11, color='darkred', horizontalalignment='right', verticalalignment='center')
    else: # 10M Samples
        plt.annotate(name, (n_times[i], n_digits[i]), xytext=(12, 22), textcoords='offset points', fontsize=11, color='darkred', horizontalalignment='left', verticalalignment='center')

# Plot NeSy Integrator (Purple)
plt.plot(ni_times, ni_digits, marker='*', markersize=15, linestyle='none', color='purple', label='Old (Native Integrator)')
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
    plt.xlim(min(all_times) * 0.5, max(all_times) * 8.0)

plot_path = os.path.abspath("pure_comparison_benchmark.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Saved plot with 1M/10M adjusted a hint left/up: {plot_path}")
