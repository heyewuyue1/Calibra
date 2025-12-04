import json
import sys
import matplotlib.pyplot as plt
import numpy as np

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def align_data(data1, data2):
    # Get the union of all query names
    keys = sorted(set(data1.keys()) | set(data2.keys()))
    values1 = [data1.get(k, 0) for k in keys]
    values2 = [data2.get(k, 0) for k in keys]
    return keys, values1, values2

def plot_bar(keys, values1, values2, label1, label2):
    x = np.arange(len(keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(keys)//3),5))
    rects1 = ax.bar(x - width/2, values1, width, label=label1)
    rects2 = ax.bar(x + width/2, values2, width, label=label2)

    ax.set_ylabel('Value')
    ax.set_title('Comparison of Values in Two JSON Files')
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=90, fontsize=8)
    ax.legend()

    fig.tight_layout()
    plt.savefig(f"/home/hejiahao/Calibra/results/{label1}_vs_{label2}.png", dpi=300)

if __name__ == "__main__":
    # Usage: python comp2json.py job_cbo.json job_l4_10it.json.json
    if len(sys.argv) != 3:
        print("Usage: python comp2json.py FILE1 FILE2")
        sys.exit(1)

    file1, file2 = sys.argv[1], sys.argv[2]
    label1 = file1.split('/')[-1].split('.')[0]
    label2 = file2.split('/')[-1].split('.')[0]

    data1 = load_json(file1)
    data2 = load_json(file2)
    keys, values1, values2 = align_data(data1, data2)
    plot_bar(keys, values1, values2, label1, label2)
    total1 = sum(values1)
    total2 = sum(values2)
    print(f"Sum of {label1}: {total1}")
    print(f"Sum of {label2}: {total2}")
    if total1 != 0:
        relative_reduce = (total1 - total2) / total1
        print(f"Relative reduction: {relative_reduce:.4f} ({relative_reduce*100:.2f}%)")
    else:
        print("Cannot compute relative reduction because the sum of the first file is zero.")
