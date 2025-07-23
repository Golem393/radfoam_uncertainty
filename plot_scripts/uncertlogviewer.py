import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
    'legend.fontsize': 14,
    'figure.titlesize': 20
})

def parse_log_data(log_filepath):
    with open(log_filepath, 'r') as f:
        log_string = f.read()

    entries = re.split(r'--- Uncertainty Metrics \[', log_string)[1:]
    parsed_data = []

    for entry in entries:
        lines = entry.strip().split('\n')
        header_line = lines[0]

        data = {}
        if 'Training epoch' in header_line:
            epoch_match = re.search(r'Training epoch(\d+)', header_line)
            data['Epoch'] = int(epoch_match.group(1)) if epoch_match else None
            data['Stage'] = f"Epoch {data['Epoch']}"
            data['Type'] = 'Training'
        elif 'Before Pruning' in header_line:
            iter_match = re.search(r'iter (\d+)', header_line)
            data['Iteration'] = int(iter_match.group(1)) if iter_match else None
            data['Stage'] = f"Before Pruning (iter {data['Iteration']})"
            data['Type'] = 'Pruning_Before'
        elif 'After Pruning' in header_line:
            iter_match = re.search(r'iter (\d+)', header_line)
            data['Iteration'] = int(iter_match.group(1)) if iter_match else None
            data['Stage'] = f"After Pruning (iter {data['Iteration']})"
            data['Type'] = 'Pruning_After'
        elif 'End of Training' in header_line:
            data['Epoch'] = 'End'
            data['Stage'] = "End of Training"
            data['Type'] = 'Training'
        else:
            print(f"Unknown entry format: {header_line}")
            data['Epoch'] = None
            data['Stage'] = "Unknown"
            data['Type'] = 'Unknown'
        metrics_section_found = False
        bin_distribution_found = False
        bin_data = {}
        for i, line in enumerate(lines):
            if 'Active Points:' in line:
                data['Active Points'] = int(line.split(':')[1].strip())
                metrics_section_found = True
            elif 'Mean:' in line and metrics_section_found:
                parts = line.split(',')
                data['Mean'] = float(parts[0].split(':')[1].strip())
                data['Std'] = float(parts[1].split(':')[1].strip())
                data['Entropy'] = float(parts[2].split(':')[1].strip())
            elif 'Quantiles:' in line and metrics_section_found:
                parts = line.split(':')
                q_values_raw = parts[1:]
                full_quantile_string = ','.join(q_values_raw)
                matches = re.findall(r'(\d+\.\d+)', full_quantile_string)
                q_values = [float(m) for m in matches]
                data['Q10'] = q_values[0]
                data['Q50'] = q_values[1]
                data['Q90'] = q_values[2]
            elif 'Low-certainty region (<0.2):' in line and metrics_section_found:
                data['Low-uncertainty region (<0.2)'] = float(line.split(':')[1].strip())
            elif 'High-certainty region (>0.8):' in line and metrics_section_found:
                data['High-uncertainty region (>0.8)'] = float(line.split(':')[1].strip())
            elif '--- Uncertainty Bin Distribution ---' in line:
                bin_distribution_found = True
            elif bin_distribution_found and line.strip() != '----------------------------------------':
                if ':' in line:
                    bin_range, percentage_str = line.strip().split(':')
                    bin_data[bin_range.strip()] = float(percentage_str.replace('%', '').strip()) / 100.0

        if bin_data:
            data['Bin Distribution'] = bin_data
        parsed_data.append(data)
    return parsed_data

def plot_training_metrics(df_training, pdf):
    epochs = df_training[df_training['Epoch'] != 'End']['Epoch'].astype(int)
    mean_values = df_training[df_training['Epoch'] != 'End']['Mean']
    std_values = df_training[df_training['Epoch'] != 'End']['Std']
    entropy_values = df_training[df_training['Epoch'] != 'End']['Entropy']

    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    fig.suptitle('Uncertainty Metrics Over Training Epochs', fontsize=16)

    axes[0].plot(epochs, mean_values, marker='o', linestyle='-', color='skyblue')
    axes[0].set_title('Mean Uncertainty')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Mean')
    axes[0].grid(True)

    axes[1].plot(epochs, std_values, marker='o', linestyle='-', color='lightcoral')
    axes[1].set_title('Standard Deviation of Uncertainty')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Std')
    axes[1].grid(True)

    axes[2].plot(epochs, entropy_values, marker='o', linestyle='-', color='lightgreen')
    axes[2].set_title('Entropy of Uncertainty')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Entropy')
    axes[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    fig.suptitle('Uncertainty Quantiles Over Training Epochs', fontsize=16)

    q10_values = df_training[df_training['Epoch'] != 'End']['Q10']
    q50_values = df_training[df_training['Epoch'] != 'End']['Q50']
    q90_values = df_training[df_training['Epoch'] != 'End']['Q90']

    axes[0].plot(epochs, q10_values, marker='o', linestyle='-', color='blue')
    axes[0].set_title('Q10 (10th Percentile) of Uncertainty')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Q10')
    axes[0].grid(True)

    axes[1].plot(epochs, q50_values, marker='o', linestyle='-', color='orange')
    axes[1].set_title('Q50 (Median) of Uncertainty')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Q50')
    axes[1].grid(True)

    axes[2].plot(epochs, q90_values, marker='o', linestyle='-', color='red')
    axes[2].set_title('Q90 (90th Percentile) of Uncertainty')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Q90')
    axes[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    fig.suptitle('Uncertainty Region Proportions Over Training Epochs', fontsize=16)

    low_certainty = df_training[df_training['Epoch'] != 'End']['Low-uncertainty region (<0.2)']
    high_certainty = df_training[df_training['Epoch'] != 'End']['High-uncertainty region (>0.8)']

    axes[0].plot(epochs, low_certainty, marker='o', linestyle='-', color='purple')
    axes[0].set_title('Average uncertainty of Low-uncertainty region (<0.2)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Average Unc')
    axes[0].grid(True)

    axes[1].plot(epochs, high_certainty, marker='o', linestyle='-', color='green')
    axes[1].set_title('Average uncertainty of High-uncertainty region (>0.8)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Average Unc')
    axes[1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)

def plot_bin_distribution(data_entry, title, pdf):
    bins = list(data_entry['Bin Distribution'].keys())
    percentages = [val * 100 for val in data_entry['Bin Distribution'].values()]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(bins, percentages, color='teal')
    ax.set_title(f'Uncertainty Bin Distribution - {title}')
    ax.set_xlabel('Uncertainty Bins')
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

def calculate_average_pruning_changes(df):
    pruning_iterations = sorted(df[df['Type'].str.contains('Pruning')]['Iteration'].unique())
    if not pruning_iterations:
        return None
    
    metrics = ['Mean', 'Std', 'Entropy', 'Q10', 'Q50', 'Q90', 
               'Low-uncertainty region (<0.2)', 'High-uncertainty region (>0.8)']
    
    changes = {metric: [] for metric in metrics}
    bin_changes = {}
    
    for iteration in pruning_iterations:
        before = df[(df['Type'] == 'Pruning_Before') & (df['Iteration'] == iteration)].iloc[0]
        after = df[(df['Type'] == 'Pruning_After') & (df['Iteration'] == iteration)].iloc[0]
        
        for metric in metrics:
            changes[metric].append(after[metric] - before[metric])
        
        if 'Bin Distribution' in before and 'Bin Distribution' in after:
            for bin_name in before['Bin Distribution'].keys():
                if bin_name not in bin_changes:
                    bin_changes[bin_name] = []
                bin_changes[bin_name].append(
                    after['Bin Distribution'][bin_name] - before['Bin Distribution'][bin_name]
                )
    
    avg_changes = {metric: np.mean(values) for metric, values in changes.items()}
    
    if bin_changes:
        avg_bin_changes = {bin_name: np.mean(values) for bin_name, values in bin_changes.items()}
        avg_changes['Bin Distribution'] = avg_bin_changes
    
    return avg_changes

def plot_average_pruning_changes(avg_changes, pdf):
    if not avg_changes:
        return
    
    metrics = ['Mean', 'Std', 'Entropy']
    values = [avg_changes[metric] for metric in metrics]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax.set_title('Average Change in Key Metrics After Pruning')
    ax.set_ylabel('Average Change (After - Before)')
    ax.grid(True, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    
    quantiles = ['Q10', 'Q50', 'Q90']
    q_values = [avg_changes[q] for q in quantiles]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(quantiles, q_values, color=['blue', 'orange', 'red'])
    ax.set_title('Average Change in Quantiles After Pruning')
    ax.set_ylabel('Average Change (After - Before)')
    ax.grid(True, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    
    regions = ['Low-uncertainty region (<0.2)', 'High-uncertainty region (>0.8)']
    r_values = [avg_changes[r] for r in regions]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(regions, r_values, color=['purple', 'green'])
    ax.set_title('Average Change in Certainty Regions After Pruning')
    ax.set_ylabel('Average Change (After - Before)')
    ax.grid(True, axis='y')
    plt.xticks(rotation=15, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    
    if 'Bin Distribution' in avg_changes:
        bins = list(avg_changes['Bin Distribution'].keys())
        changes = [val * 100 for val in avg_changes['Bin Distribution'].values()]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(bins, changes, color='teal')
        ax.set_title('Average Change in Uncertainty Bin Distribution After Pruning')
        ax.set_xlabel('Uncertainty Bins')
        ax.set_ylabel('Average Percentage Change (After - Before)')
        ax.grid(True, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

def create_pdf_report(log_filepath, output_pdf_path='uncertainty_report.pdf'):
    parsed_data = parse_log_data(log_filepath)
    df = pd.DataFrame(parsed_data)

    with PdfPages(output_pdf_path) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.text(0.1, 0.9, "Uncertainty Metrics Analysis Report", fontsize=20, fontweight='bold')
        ax.text(0.1, 0.8, f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", fontsize=12)
        ax.text(0.1, 0.7, "Contents:", fontsize=14, fontweight='bold')
        ax.text(0.15, 0.65, "1. Overall Training Progress", fontsize=12)
        ax.text(0.15, 0.60, "2. Uncertainty Bin Distributions During Training", fontsize=12)
        ax.text(0.15, 0.55, "3. Before vs. After Pruning Analysis", fontsize=12)
        ax.text(0.15, 0.50, "4. Average Pruning Effects", fontsize=12)
        ax.text(0.15, 0.45, "5. Final Training State Analysis", fontsize=12)
        ax.axis('off')
        pdf.savefig(fig)
        plt.close(fig)

        pdf.attach_note("Section 1: Overall Training Progress")
        training_df = df[df['Type'] == 'Training'].copy()
        
        training_df['Numeric_Epoch'] = training_df['Epoch'].apply(
            lambda x: int(x) if isinstance(x, (int, str)) and x != 'End'
            else (
                df['Epoch'][df['Epoch'].apply(lambda val: isinstance(val, (int, str)) and val != 'End')].astype(int).max() + 500
                if x == 'End'
                else None
            )
        )
        training_df = training_df.sort_values(by='Numeric_Epoch').reset_index(drop=True)

        plot_training_metrics(training_df, pdf)

        pdf.attach_note("Section 2: Uncertainty Bin Distributions During Training")
        
        max_valid_epoch = df['Epoch'].apply(lambda x: int(x) if isinstance(x, (int, str)) and x != 'End' else -1).max()

        selected_epochs_for_bins = []
        
        if 'End' in df['Epoch'].values and max_valid_epoch != 'End':
            if max_valid_epoch != -1:
                selected_epochs_for_bins.append('End')
        
        selected_epochs_for_bins = [e for e in selected_epochs_for_bins if e is not None and e in training_df['Epoch'].values]
        
        end_of_training_entry = df[(df['Type'] == 'Training') & (df['Epoch'] == 'End')]
        if not end_of_training_entry.empty and 'End' not in selected_epochs_for_bins:
             selected_epochs_for_bins.append('End')


        for epoch_val in selected_epochs_for_bins:
            entry = training_df[training_df['Epoch'] == epoch_val].iloc[0]
            plot_bin_distribution(entry, entry['Stage'], pdf)

        pdf.attach_note("Section 3: Before vs. After Pruning Analysis")
        pruning_iterations = sorted(df[df['Type'].str.contains('Pruning')]['Iteration'].unique())

        for iteration in pruning_iterations:
            before_pruning = df[(df['Type'] == 'Pruning_Before') & (df['Iteration'] == iteration)].iloc[0]
            after_pruning = df[(df['Type'] == 'Pruning_After') & (df['Iteration'] == iteration)].iloc[0]

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Metrics Before vs. After Pruning (Iteration {iteration})', fontsize=16)

            metrics = ['Mean', 'Std', 'Entropy']
            labels = ['Before Pruning', 'After Pruning']
            colors = ['skyblue', 'salmon']

            for i, metric in enumerate(metrics):
                values = [before_pruning[metric], after_pruning[metric]]
                axes[i].bar(labels, values, color=colors)
                axes[i].set_title(metric)
                axes[i].set_ylabel(metric)
                for j, v in enumerate(values):
                    axes[i].text(j, v + (0.01 if v < 0.5 else -0.01), f'{v:.4f}', ha='center', va='bottom' if v < 0.5 else 'top')

            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            pdf.savefig(fig)
            plt.close(fig)

            fig, axes = plt.subplots(1, 2, figsize=(18, 7))
            fig.suptitle(f'Uncertainty Bin Distribution Before vs. After Pruning (Iteration {iteration})', fontsize=16)

            bins = list(before_pruning['Bin Distribution'].keys())
            before_percentages = [val * 100 for val in before_pruning['Bin Distribution'].values()]
            after_percentages = [val * 100 for val in after_pruning['Bin Distribution'].values()]

            axes[0].bar(bins, before_percentages, color='skyblue')
            axes[0].set_title('Before Pruning')
            axes[0].set_xlabel('Uncertainty Bins')
            axes[0].set_ylabel('Percentage (%)')
            axes[0].set_ylim(0, 100)
            axes[0].tick_params(axis='x', rotation=45)

            axes[1].bar(bins, after_percentages, color='salmon')
            axes[1].set_title('After Pruning')
            axes[1].set_xlabel('Uncertainty Bins')
            axes[1].set_ylabel('Percentage (%)')
            axes[1].set_ylim(0, 100)
            axes[1].tick_params(axis='x', rotation=45)

            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            pdf.savefig(fig)
            plt.close(fig)

        pdf.attach_note("Section 4: Average Pruning Effects")
        avg_changes = calculate_average_pruning_changes(df)
        
        if avg_changes:
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_subplot(111)
            ax.text(0.1, 0.9, "Average Pruning Effects Summary", fontsize=18, fontweight='bold')
            
            if pruning_iterations:
                ax.text(0.1, 0.85, f"Based on {len(pruning_iterations)} pruning iterations", fontsize=12)
            
            y_pos = 0.8
            for metric, change in avg_changes.items():
                if metric != 'Bin Distribution':
                    ax.text(0.1, y_pos, f"{metric}: {change:.4f}", fontsize=12)
                    y_pos -= 0.05
            
            ax.axis('off')
            pdf.savefig(fig)
            plt.close(fig)
            
            plot_average_pruning_changes(avg_changes, pdf)
        pdf.attach_note("Section 5: Final Training State Analysis")
        final_entry = df[df['Stage'] == 'End of Training'].iloc[0]

        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.text(0.1, 0.9, "Final Training State Uncertainty Metrics", fontsize=18, fontweight='bold')
        ax.text(0.1, 0.8, f"Active Points: {final_entry['Active Points']}", fontsize=12)
        ax.text(0.1, 0.75, f"Mean: {final_entry['Mean']:.4f}", fontsize=12)
        ax.text(0.1, 0.70, f"Std: {final_entry['Std']:.4f}", fontsize=12)
        ax.text(0.1, 0.65, f"Entropy: {final_entry['Entropy']:.4f}", fontsize=12)
        ax.text(0.1, 0.60, f"Quantiles: Q10={final_entry['Q10']:.4f}, Q50={final_entry['Q50']:.4f}, Q90={final_entry['Q90']:.4f}", fontsize=12)
        ax.text(0.1, 0.55, f"Low-uncertainty region (<0.2): {final_entry['Low-uncertainty region (<0.2)']:.4f}", fontsize=12)
        ax.text(0.1, 0.50, f"High-uncertainty region (>0.8): {final_entry['High-uncertainty region (>0.8)']:.4f}", fontsize=12)

        ax.axis('off')
        pdf.savefig(fig)
        plt.close(fig)

        plot_bin_distribution(final_entry, "End of Training", pdf)

    print(f"PDF report '{output_pdf_path}' generated successfully.")

if __name__ == "__main__":
    log_file_name = 'output_final/normal_setting/uncertainty_log.txt
    create_pdf_report(log_file_name)