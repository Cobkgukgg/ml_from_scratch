"""Generate a markdown benchmark report and save to docs/benchmark_results.md."""

import io, sys
sys.stdout = buf = io.StringIO()

try:
    import benchmark_vs_sklearn
    benchmark_vs_sklearn.main()
except Exception as e:
    print(f"Error: {e}")

sys.stdout = sys.__stdout__
output = buf.getvalue()
print(output)

with open("../docs/benchmark_results.md", "w") as f:
    f.write("# Benchmark Results\n\n```\n")
    f.write(output)
    f.write("\n```\n")
print("Saved to docs/benchmark_results.md")
