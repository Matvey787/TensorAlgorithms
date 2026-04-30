import json
import matplotlib.pyplot as plt

import sys
from collections import defaultdict
import numpy as np

def plot_performance_ndjson(json_file, parameter):
    groups = defaultdict(lambda: {"naive": [], 
                                  "winograd": [], 
                                  "im2col": [], 
                                  "speedup_Winograd": [], 
                                  "speedup_Im2Col": []})

    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                item = json.loads(line)
                if "input" not in item: continue
                
                x_value = item["input"].get(parameter)
                if x_value is None: continue

                groups[x_value]["naive"].append(item.get("naive_ms", 0))
                groups[x_value]["winograd"].append(item.get("winograd_ms", 0))
                groups[x_value]["im2col"].append(item.get("im2col_ms", 0))
                groups[x_value]["speedup_Winograd"].append(item.get("speedupWinograd", 0))
                groups[x_value]["speedup_Im2Col"].append(item.get("speedupIm2Col", 0))

            except json.JSONDecodeError:
                continue


    sorted_x = sorted(groups.keys())
    x_arr = np.array(sorted_x)
    naive_avg = np.array([np.mean(groups[x]["naive"]) for x in sorted_x])
    wino_avg = np.array([np.mean(groups[x]["winograd"]) for x in sorted_x])
    im2col_avg = np.array([np.mean(groups[x]["im2col"]) for x in sorted_x])

    speedupWin_avg = np.array([np.mean(groups[x]["speedup_Winograd"]) for x in sorted_x])
    mean_speedupWin = float(np.mean(speedupWin_avg))

    speedupIm2Col_avg = np.array([np.mean(groups[x]["speedup_Im2Col"]) for x in sorted_x])
    mean_speedupIm2Col = float(np.mean(speedupIm2Col_avg))


    plt.figure(figsize=(16, 7))






    ax1 = plt.subplot(1, 2, 1)
    
    ax1.plot(x_arr, naive_avg, 'o-', color='red', alpha=0.3, markersize=4, label='Naive')
    ax1.plot(x_arr, wino_avg, 'o-', color='blue', alpha=0.3, markersize=4, label='Winograd')
    ax1.plot(x_arr, im2col_avg, 'o-', color='green', alpha=0.3, markersize=4, label='Im2Col')

    x_smooth = np.linspace(x_arr.min(), x_arr.max(), 100)
    degree = 2 
    
    z_n = np.polyfit(x_arr, naive_avg, degree)
    p_n = np.poly1d(z_n)
    ax1.plot(x_smooth, p_n(x_smooth), 'r--', linewidth=2, label='Naive Trend')

    z_w = np.polyfit(x_arr, wino_avg, degree)
    p_w = np.poly1d(z_w)
    ax1.plot(x_smooth, p_w(x_smooth), 'b--', linewidth=2, label='Winograd Trend')

    z_i = np.polyfit(x_arr, im2col_avg, degree)
    p_i = np.poly1d(z_i)
    ax1.plot(x_smooth, p_i(x_smooth), 'g--', linewidth=2, label='Im2Col Trend')

    ax1.set_title(f'Execution Time by {parameter}', fontsize=14)
    ax1.set_xlabel(parameter)
    ax1.set_ylabel('Time (ms)')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)







    ax2 = plt.subplot(1, 2, 2)
    
    ax2.plot(x_arr, speedupWin_avg, 'b-o', markersize=4, alpha=0.4, label='Speedup Winograd')
    ax2.plot(x_arr, speedupIm2Col_avg, 'g-o', markersize=4, alpha=0.4, label='Speedup Im2Col')
    
    ax2.axhline(y=mean_speedupWin, color='darkblue', linestyle='--', linewidth=2, 
                label=f'Average Speedup: {mean_speedupWin:.2f}x')
    
    ax2.axhline(y=mean_speedupIm2Col, color='darkgreen', linestyle='--', linewidth=2, 
                label=f'Average Speedup: {mean_speedupIm2Col:.2f}x')
    
    ax2.set_title('Speedup Factor', fontsize=14)
    ax2.set_xlabel(parameter)
    ax2.set_ylabel('Speedup (x)')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot.py <file.ndjson> <tensor option>")
        sys.exit(1)

    json_path = sys.argv[1]
    param = sys.argv[2]
    plot_performance_ndjson(json_path, param)
