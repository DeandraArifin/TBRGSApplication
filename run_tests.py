import subprocess
import time
import psutil 
import pandas as pd
from tabulate import tabulate

#TO USE:
#Comment out "newg, pos = loadproblem.todraw(nodes, edges)" L.382 in main, and uncomment "pass"
#Comment out "loadproblem.draw(newg, pos, colour_map)" L.420 in main

methods = ['DFS', 'BFS', 'UCS', 'GBFS', 'Astar']
testcases = [
    'test_cyclic.txt',
    'test_deadend.txt',
    'test_disconnected_subgraphs.txt',
    'test_easy_single.txt',
    'test_equidistant.txt',
    'test_grid_style.txt',
    'test_long_linear.txt',
    'test_loops_multiple_paths.txt',
    'test_multiple_destinations.txt',
    'test_unreachable.txt'
]

Results = []

print("Automation script started")

for index, current_case in enumerate(testcases):
    print(f"\nTest {index + 1}: {current_case}")
    
    for method in methods:
        print(f"\nRunning {method}...")

        # Start the subprocess
        process = subprocess.Popen(
            ["python", "search.py", f'./specific_tests/{current_case}', method],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Monitor memory usage while process is running
        mem_usage = []
        start_time = time.time()

        proc = psutil.Process(process.pid)

        try:
            while process.poll() is None:
                mem_info = proc.memory_info()
                mem_usage.append(mem_info.rss / (1024 * 1024))  # Convert to MiB
                time.sleep(0.001)  # Sampling interval
        except psutil.NoSuchProcess:
            pass  # Process ended before we could sample

        end_time = time.time()

        stdout, stderr = process.communicate()

        elapsed_time = end_time - start_time
        max_memory = max(mem_usage) if mem_usage else 0

        if "No path found" not in stdout:
            print("Path Found")
        else:
            print("No Path Found")


        lines = stdout.strip().split('\n')
        important_lines = [line for line in lines if "goal =" in line or "->" in line or "No path found" in line]
        summary = " | ".join(important_lines)

        # Store result
        Results.append({
            "testcase": current_case,
            "method": method,
            "time": elapsed_time,
            "memory": max_memory,
            "output": summary
        })

# Convert results to DataFrame
df = pd.DataFrame(Results)

# Clean up column width for output
pd.set_option('display.max_colwidth', 50)  # Limit output column width

print("\nSummary of Results:\n")
print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))

# Calculate averages per method
averages = df.groupby('method')[['time', 'memory']].mean().reset_index()

print("\nAverage Time and Memory Usage per Method:\n")
print(tabulate(averages, headers='keys', tablefmt='pretty', showindex=False))