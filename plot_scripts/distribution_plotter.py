import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def parse_distribution_log(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    iterations = re.split(r'iter (\d+)', content)[1:]
    data = []
    
    for i in range(0, len(iterations), 2):
        iter_num = int(iterations[i])
        text = iterations[i+1]
        
        distributions = re.findall(r'(.*? distribution:)\n(.*?)(?=\n\n|$)', text, re.DOTALL)
        for name, values_text in distributions:
            bins_counts = re.findall(r'\[(.*?), (.*?)\): (\d+)', values_text)
            for (low, high, count) in bins_counts:
                data.append({
                    'iter': iter_num,
                    'metric': name.strip(),
                    'bin': f"[{low}, {high})",
                    'count': int(count)
                })
    
    return pd.DataFrame(data)

def plot_metric_trends(df, metric_name, normalize=False, pdf=None):
    df_metric = df[df['metric'] == metric_name].copy()
    
    df_pivot = df_metric.pivot(index='iter', columns='bin', values='count')

    def get_lower_bound(bin_str):
        low = float(bin_str.split(',')[0][1:])
        return low
    
    sorted_bins = sorted(df_pivot.columns, key=get_lower_bound)
    df_pivot = df_pivot[sorted_bins]
    
    if normalize:
        df_pivot = df_pivot.div(df_pivot.sum(axis=1), axis=0) * 100

    plt.figure(figsize=(10, 6))
    df_pivot.plot.area(stacked=True, ax=plt.gca())
    plt.title(f'{metric_name} {"(%)" if normalize else "(count)"} over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Percentage' if normalize else 'Count')
    plt.legend(title='Bin ranges', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if pdf:
        pdf.savefig()
        plt.close()
    else:
        plt.show()

filename = 'distribution_log2.txt'
df = parse_distribution_log(filename)

with PdfPages('distribution_trends.pdf') as pdf:
    print(df['metric'])
    for metric in df['metric'].unique():
        print(df['metric'].unique())
        plot_metric_trends(df, metric, normalize=True, pdf=pdf)
        plot_metric_trends(df, metric, normalize=False, pdf=pdf)
    
    d = pdf.infodict()
    d['Title'] = 'Distribution Trends Over Iterations'
    d['Author'] = 'Flavio Arrigoni'