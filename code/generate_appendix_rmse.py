import os
import tempfile
import subprocess
import glob
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'matplotlib-cache'))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================
PROJECT_ROOT = os.getcwd()
INPUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'oos_forecasts_2000_2005')
OUT_DIR = os.path.join(INPUT_DIR, 'rmse_tables_figures')
os.makedirs(OUT_DIR, exist_ok=True)

# Map internal model IDs to publication-ready titles
MODEL_LABELS = {
    'DFM_SINGLE': 'DFM Single',
    'DFM_SINGLE_CORR': 'DFM Single (Correlation)',
    'DFM_SINGLE_EN': 'DFM Single (Elastic Net)',
    'DFM_BLOCK': 'DFM Block',
    'DFM_BLOCK_CORR': 'DFM Block (Correlation)',
    'DFM_BLOCK_EN': 'DFM Block (Elastic Net)',
    'DFM_FULLVAR': 'DFM Full VAR',
    'DFM_FULLVAR_CORR': 'DFM Full VAR (Correlation)',
    'DFM_FULLVAR_EN': 'DFM Full VAR (Elastic Net)'
}

TARGET_MODELS = list(MODEL_LABELS.keys())

REGION_NAMES = {
    'ID': 'Indonesia', 
    'MA': 'Malaysia', 
    'PH': 'Philippines', 
    'TH': 'Thailand', 
    'VT': 'Vietnam', 
    'ASEAN': 'ASEAN Aggregate'
}

COUNTRY_COLORS = {
    'Indonesia': '#1f77b4', 
    'Malaysia': '#ff7f0e', 
    'Philippines': '#2ca02c', 
    'Thailand': '#d62728', 
    'Vietnam': '#9467bd', 
    'ASEAN Aggregate': '#8c564b'
}

master_tex_lines = [
    r"\documentclass{article}",
    r"\usepackage{booktabs}",
    r"\usepackage[margin=1in]{geometry}",
    r"\begin{document}"
]

def dataframe_to_latex(df, column_format):
    """Render a simple booktabs table without pandas/Jinja2."""
    lines = [f"\\begin{{tabular}}{{{column_format}}}", "\\toprule"]
    lines.append(" & ".join(map(str, df.columns)) + r" \\")
    lines.append("\\midrule")
    for row in df.itertuples(index=False, name=None):
        lines.append(" & ".join(map(str, row)) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)

# ==========================================
# 2. OVERALL RMSE TABLES
# ==========================================
print("Processing Overall RMSE Tables...")
try:
    over = pd.read_csv(os.path.join(INPUT_DIR, 'RMSE_Overall.csv'))
    ar_over = over[over['ModelID'] == 'AR'][['Country', 'RMSE']].rename(columns={'RMSE': 'AR_RMSE'})
    
    for model in TARGET_MODELS:
        if model not in over['ModelID'].unique():
            print(f"  -> Skipping {model} (Not found in RMSE_Overall.csv)")
            continue
            
        sub = over[over['ModelID'] == model].copy()
        sub = pd.merge(sub, ar_over, on='Country', how='inner')
        
        # Calculate Relative RMSE (Model / Benchmark)
        sub['Relative_RMSE'] = sub['RMSE'] / sub['AR_RMSE']
        sub['Beats AR?'] = np.where(sub['Relative_RMSE'] < 1.0, 'Yes', 'No')
        sub['Region'] = sub['Country'].map(REGION_NAMES)
        
        # Ensure all 6 regions statically report
        master_regions = pd.DataFrame({'Region': list(REGION_NAMES.values())})
        sub = master_regions.merge(sub, on='Region', how='left')
        
        # Clean formatting
        sub.loc[sub['RMSE'].isna(), 'Beats AR?'] = '-'
        sub['AR_RMSE'] = sub['AR_RMSE'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        sub['RMSE'] = sub['RMSE'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        sub['Relative_RMSE'] = sub['Relative_RMSE'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
        
        sub = sub[['Region', 'AR_RMSE', 'RMSE', 'Relative_RMSE', 'Beats AR?']]
        sub.rename(columns={'AR_RMSE': 'AR Benchmark RMSE', 'RMSE': f'{MODEL_LABELS[model]} RMSE', 'Relative_RMSE': 'Relative RMSE'}, inplace=True)
        
        tex_sub = dataframe_to_latex(sub, 'lcccc')
        
        # Inject the mathematical footnote
        footnote = (
            "\\vspace{0.5em}\n\\small\\textit{Notes:} Headline overall RMSE is calculated across forecast horizons $h=0$ to $h=4$ and excludes $h=-1$ backcasts. Backcast RMSE is reported separately in \\texttt{RMSE\\_Backcast.csv}. "
            "The ASEAN Aggregate error is calculated using dynamic, time-varying GDP weights before the final temporal RMSE aggregation."
        )
        
        table_env = (
            "\\begin{table}[htbp]\n\\centering\n"
            f"\\caption{{Overall Root Mean Squared Error (RMSE): {MODEL_LABELS[model]}}}\n"
            f"{tex_sub}\n"
            f"{footnote}\n"
            "\\end{table}\n\n"
        )
        
        with open(os.path.join(OUT_DIR, f'Table_RMSE_Overall_{model}_snippet.tex'), 'w') as f:
            f.write(table_env)
            
        master_tex_lines.append(table_env)
except FileNotFoundError:
    print("Warning: RMSE_Overall.csv not found.")

# ==========================================
# 3. EVENT WINDOW LINE CHARTS
# ==========================================
print("Generating Event Window Charts...")
try:
    ew = pd.read_csv(os.path.join(INPUT_DIR, 'RMSE_by_EventWindow.csv'))
    ar_ew = ew[ew['ModelID'] == 'AR'][['Country', 'MonthDiff', 'RMSE']].rename(columns={'RMSE': 'AR_RMSE'})
    
    for model in TARGET_MODELS:
        if model not in ew['ModelID'].unique():
            continue
            
        m_ew = ew[ew['ModelID'] == model].copy()
        merged = pd.merge(m_ew, ar_ew, on=['Country', 'MonthDiff'], how='inner')
        
        # Strict filtering to -12 to 2 months
        merged = merged[merged['MonthDiff'].between(-12, 2)].copy()
        merged['Relative_RMSE'] = merged['RMSE'] / merged['AR_RMSE']
        
        fig, ax = plt.subplots(figsize=(9, 5.5))
        
        for country in REGION_NAMES.keys():
            c_sub = merged[merged['Country'] == country].sort_values('MonthDiff')
            if c_sub.empty: continue
            
            region_name = REGION_NAMES[country]
            color = COUNTRY_COLORS.get(region_name, 'black')
            ax.plot(c_sub['MonthDiff'], c_sub['Relative_RMSE'], marker='o', markersize=5, linewidth=2.5, label=region_name, color=color)
            
        # Add parity line (1.0)
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, label='AR Benchmark Parity (1.0)')
        
        ax.set_title(f'Relative RMSE Across Event Window\n{MODEL_LABELS[model]} vs. AR Benchmark', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Months relative to target quarter-end', fontsize=11)
        
        # Updated Y-Axis Label
        ax.set_ylabel('RMSE Ratio (DFM / AR)', fontsize=11)
        
        ax.set_ylim(bottom=0)
        ax.set_xlim(-12, 2)
        ax.set_xticks(np.arange(-12, 3, 2))
        ax.grid(True, linestyle=':', alpha=0.6)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'RMSE_EventWindow_{model}.png'), dpi=300)
        plt.close()
except FileNotFoundError:
    print("Warning: RMSE_by_EventWindow.csv not found.")

# ==========================================
# 4. HORIZON LINE CHARTS
# ==========================================
print("Generating Horizon Charts...")
try:
    hz = pd.read_csv(os.path.join(INPUT_DIR, 'RMSE_by_Horizon.csv'))
    ar_hz = hz[hz['ModelID'] == 'AR'][['Country', 'HorizonQ', 'RMSE']].rename(columns={'RMSE': 'AR_RMSE'})
    
    for model in TARGET_MODELS:
        if model not in hz['ModelID'].unique():
            continue
            
        m_hz = hz[hz['ModelID'] == model].copy()
        merged = pd.merge(m_hz, ar_hz, on=['Country', 'HorizonQ'], how='inner')
        
        merged = merged[merged['HorizonQ'].between(0, 4)].copy()
        merged['Relative_RMSE'] = merged['RMSE'] / merged['AR_RMSE']
        
        fig, ax = plt.subplots(figsize=(8, 5.5))
        
        for country in REGION_NAMES.keys():
            c_sub = merged[merged['Country'] == country].sort_values('HorizonQ')
            if c_sub.empty: continue
            
            region_name = REGION_NAMES[country]
            color = COUNTRY_COLORS.get(region_name, 'black')
            ax.plot(c_sub['HorizonQ'], c_sub['Relative_RMSE'], marker='o', markersize=6, linewidth=2.5, label=region_name, color=color)
            
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, label='AR Benchmark Parity (1.0)')
        
        ax.set_title(f'Relative RMSE by Forecast Horizon\n{MODEL_LABELS[model]} vs. AR Benchmark', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Forecast Horizon (Quarters Ahead)', fontsize=11)
        
        # Updated Y-Axis Label
        ax.set_ylabel('RMSE Ratio (DFM / AR)', fontsize=11)
        
        ax.set_ylim(bottom=0)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.grid(True, linestyle=':', alpha=0.6)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'RMSE_Horizon_{model}.png'), dpi=300)
        plt.close()
except FileNotFoundError:
    print("Warning: RMSE_by_Horizon.csv not found.")

# ==========================================
# 5. COMPILE MASTER PDF FOR TABLES
# ==========================================
print("Compiling Master PDF of all tables...")
master_tex_lines.append(r"\end{document}")
master_tex_path = os.path.join(OUT_DIR, 'Master_Appendix_RMSE.tex')

with open(master_tex_path, 'w') as f:
    f.write("\n".join(master_tex_lines))

try:
    subprocess.run(
        ['pdflatex', '-interaction=nonstopmode', 'Master_Appendix_RMSE.tex'],
        cwd=OUT_DIR, 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL
    )
    # Cleanup aux files
    for ext in ['*.aux', '*.log']:
        for file in glob.glob(os.path.join(OUT_DIR, ext)):
            os.remove(file)
except FileNotFoundError:
    print("Warning: pdflatex not found on system path.")

print(f"Success! All charts, tex snippets, and the master PDF are saved in:\n{OUT_DIR}")
