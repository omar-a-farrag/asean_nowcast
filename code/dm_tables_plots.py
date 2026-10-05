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
OUT_DIR = os.path.join(INPUT_DIR, 'dm_tables_figures')
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

# Map Region Names
REGION_NAMES = {
    'ID': 'Indonesia', 
    'MA': 'Malaysia', 
    'PH': 'Philippines', 
    'TH': 'Thailand', 
    'VT': 'Vietnam', 
    'ASEAN': 'ASEAN Aggregate'
}

# Color palette mapped to explicit names
COUNTRY_COLORS = {
    'Indonesia': '#1f77b4', 
    'Malaysia': '#ff7f0e', 
    'Philippines': '#2ca02c', 
    'Thailand': '#d62728', 
    'Vietnam': '#9467bd', 
    'ASEAN Aggregate': '#8c564b'
}

# Master list to collect LaTeX strings for the compiled PDF
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
# 2. LOAD THE AUTHORITATIVE MATLAB HAC-DM OUTPUTS
# ==========================================
print("Loading target-level Newey-West/HAC DM results from the evaluator...")

try:
    over = pd.read_csv(os.path.join(INPUT_DIR, 'DM_Test_Overall.csv'))
    horizon = pd.read_csv(os.path.join(INPUT_DIR, 'DM_Test_by_Horizon.csv'))
    event = pd.read_csv(os.path.join(INPUT_DIR, 'DM_Test_by_EventWindow.csv'))
except FileNotFoundError as exc:
    raise SystemExit(
        f"Missing evaluator DM output: {exc.filename}. Run "
        "asean_nowcast_evaluator2.m first."
    )

required_common = {
    'Country', 'ModelID', 'DM_Stat', 'P_Value', 'Significance',
    'Winner', 'N_Targets', 'HAC_Lag', 'Mean_Loss_Diff'
}
for name, frame, extra in [
    ('overall', over, set()),
    ('horizon', horizon, {'HorizonQ'}),
    ('event-window', event, {'MonthDiff'}),
]:
    missing = (required_common | extra) - set(frame.columns)
    if missing:
        raise SystemExit(
            f"{name} DM file is not the corrected HAC version; missing: "
            + ', '.join(sorted(missing))
        )
print(" -> Loaded authoritative HAC-DM outputs; no statistics were recomputed.")

# ==========================================
# 3. LOCKED HYPERPARAMETERS TABLE
# ==========================================
print("Processing Locked Hyperparameters...")
try:
    hp = pd.read_csv(os.path.join(INPUT_DIR, 'Locked_Hyperparameters.csv'))
    hp = hp.drop(columns=['DFM_Dom_Factors'], errors='ignore')
    
    # Map Country acronyms to Full Regions
    hp['Region'] = hp['Country'].map(REGION_NAMES)
    
    # Merge against a Master Frame to guarantee all 6 regions report
    master_regions = pd.DataFrame({'Region': list(REGION_NAMES.values())})
    hp = master_regions.merge(hp, on='Region', how='left')
    
    hp = hp[['Region', 'AR_Lags', 'DFM_Single_Factors', 'DFM_Glob_Factors']]
    hp.rename(columns={
        'AR_Lags': 'AR Lags',
        'DFM_Single_Factors': 'DFM Single Factors',
        'DFM_Glob_Factors': 'DFM Full VAR Factors (Global)'
    }, inplace=True)
    
    # Replace NaNs (like ASEAN) with dashes
    hp = hp.fillna('-')
    
    # Format floats into integers where possible
    for col in hp.columns:
        if col != 'Region':
            hp[col] = hp[col].apply(lambda x: str(int(x)) if isinstance(x, float) and pd.notna(x) else str(x))
            
    tex_hp = dataframe_to_latex(hp, 'lccc')
    
    table_env = (
        "\\begin{table}[htbp]\n\\centering\n"
        "\\caption{Locked Hyperparameters by Region}\n"
        f"{tex_hp}\n"
        "\\end{table}\n\n"
    )
    
    with open(os.path.join(OUT_DIR, 'Table_Hyperparameters_snippet.tex'), 'w') as f:
        f.write(table_env)
        
    master_tex_lines.append(table_env)
except FileNotFoundError:
    print("Warning: Locked_Hyperparameters.csv not found.")

# ==========================================
# 4. DM TEST OVERALL TABLES
# ==========================================
print("Processing DM Test Overall Tables...")
master_tex_lines.append(r"\clearpage")

for model in TARGET_MODELS:
    if over.empty or model not in over['ModelID'].unique():
        continue
        
    sub = over[over['ModelID'] == model].copy()
    
    sub['Beats AR?'] = np.where(sub['Winner'] == model, 'Yes', 'No')
    sub['Region'] = sub['Country'].map(REGION_NAMES)
    
    # Ensure all 6 regions are statically mapped onto the table
    master_regions = pd.DataFrame({'Region': list(REGION_NAMES.values())})
    sub = master_regions.merge(sub, on='Region', how='left')
    
    # Clean formatting and map NaNs (for uncalculated rows) to dashes
    sub.loc[sub['DM_Stat'].isna(), 'Beats AR?'] = '-'
    sub['DM_Stat'] = sub['DM_Stat'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    sub['P_Value'] = sub['P_Value'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
    sub['Significance'] = sub['Significance'].fillna('')
    
    sub = sub[['Region', 'DM_Stat', 'P_Value', 'Significance', 'Beats AR?']]
    sub.rename(columns={'DM_Stat': 'DM Statistic', 'P_Value': 'p-value'}, inplace=True)
    
    tex_sub = dataframe_to_latex(sub, 'lcccc')
    
    table_env = (
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{Diebold-Mariano Overall Test Results: {MODEL_LABELS[model]}}}\n"
        f"{tex_sub}\n"
        "\\end{table}\n\n"
    )
    
    with open(os.path.join(OUT_DIR, f'Table_DM_Overall_{model}_snippet.tex'), 'w') as f:
        f.write(table_env)
        
    master_tex_lines.append(table_env)

# ==========================================
# 5. EVENT WINDOW LINE CHARTS
# ==========================================
print("Generating Event Window Charts...")
if not event.empty:
    for model in TARGET_MODELS:
        if model not in event['ModelID'].unique():
            continue
            
        sub = event[event['ModelID'] == model]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        
        for country in REGION_NAMES.keys():
            c_sub = sub[sub['Country'] == country].sort_values('MonthDiff')
            if c_sub.empty:
                continue
            region_name = REGION_NAMES[country]
            color = COUNTRY_COLORS.get(region_name, 'black')
            ax.plot(c_sub['MonthDiff'], c_sub['P_Value'], marker='o', markersize=5, linewidth=2.5, label=region_name, color=color)
            
        # Add exact Significance Threshold lines
        ax.axhline(0.10, color='gray', linestyle='--', linewidth=1.5, label='10% Significance')
        ax.axhline(0.05, color='orange', linestyle='--', linewidth=1.5, label='5% Significance')
        ax.axhline(0.01, color='red', linestyle='--', linewidth=1.5, label='1% Significance')
        
        ax.set_title(f'Diebold-Mariano Test p-values Across Event Window\n{MODEL_LABELS[model]}', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Months relative to target quarter-end', fontsize=11)
        ax.set_ylabel('p-value', fontsize=11)
        
        ax.set_ylim(bottom=0)
        ax.set_xlim(-12, 2)
        
        ax.set_xticks(np.arange(-12, 3, 2))
        ax.grid(True, linestyle=':', alpha=0.6)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'DM_EventWindow_{model}.png'), dpi=300)
        plt.close()

# ==========================================
# 6. HORIZON LINE CHARTS
# ==========================================
print("Generating Horizon Charts...")
if not horizon.empty:
    for model in TARGET_MODELS:
        if model not in horizon['ModelID'].unique():
            continue
            
        sub = horizon[horizon['ModelID'] == model]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        
        for country in REGION_NAMES.keys():
            c_sub = sub[sub['Country'] == country].sort_values('HorizonQ')
            if c_sub.empty:
                continue
            region_name = REGION_NAMES[country]
            color = COUNTRY_COLORS.get(region_name, 'black')
            ax.plot(c_sub['HorizonQ'], c_sub['P_Value'], marker='o', markersize=6, linewidth=2.5, label=region_name, color=color)
            
        ax.axhline(0.10, color='gray', linestyle='--', linewidth=1.5, label='10% Significance')
        ax.axhline(0.05, color='orange', linestyle='--', linewidth=1.5, label='5% Significance')
        ax.axhline(0.01, color='red', linestyle='--', linewidth=1.5, label='1% Significance')
        
        ax.set_title(f'Diebold-Mariano Test p-values by Horizon\n{MODEL_LABELS[model]}', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Forecast Horizon (Quarters Ahead)', fontsize=11)
        ax.set_ylabel('p-value', fontsize=11)
        
        ax.set_ylim(bottom=0)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.grid(True, linestyle=':', alpha=0.6)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'DM_Horizon_{model}.png'), dpi=300)
        plt.close()

# ==========================================
# 7. COMPILE MASTER PDF FOR TABLES
# ==========================================
print("Compiling Master PDF of all tables...")
master_tex_lines.append(r"\end{document}")
master_tex_path = os.path.join(OUT_DIR, 'Master_Appendix_Tables.tex')

with open(master_tex_path, 'w') as f:
    f.write("\n".join(master_tex_lines))

try:
    subprocess.run(
        ['pdflatex', '-interaction=nonstopmode', 'Master_Appendix_Tables.tex'],
        cwd=OUT_DIR, 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL
    )
    for ext in ['*.aux', '*.log']:
        for file in glob.glob(os.path.join(OUT_DIR, ext)):
            os.remove(file)
except FileNotFoundError:
    print("Warning: pdflatex not found on system path.")

print(f"Success! All charts, tex snippets, and the master PDF are saved in:\n{OUT_DIR}")
