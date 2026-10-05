import pandas as pd
import numpy as np
import os
import subprocess
import glob

# ==========================================
# 1. Configuration & File Paths
# ==========================================
PROJECT_ROOT = os.getcwd()

# Support both 'spec' and 'specs' folder names
SPEC_DIR = os.path.join(PROJECT_ROOT, 'input', 'spec')
if not os.path.exists(SPEC_DIR):
    SPEC_DIR = os.path.join(PROJECT_ROOT, 'input', 'specs')
    
SPEC_FILE = os.path.join(SPEC_DIR, "spec_asean.xlsx")
DATA_DIR = os.path.join(PROJECT_ROOT, 'input', 'data')

OUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'oos_forecasts_2000_2005', 'summary_tables')
os.makedirs(OUT_DIR, exist_ok=True)

# Resolve the as-of date from the audited news-model configuration. Fall
# back to the latest date for which all five country files exist.
AUDIT_FILE = os.path.join(
    PROJECT_ROOT, 'output', 'oos_forecasts_2000_2005',
    'decomposition', 'News_Model_Audit.csv'
)
if os.path.exists(AUDIT_FILE):
    audit_df = pd.read_csv(AUDIT_FILE)
    asof_values = audit_df['DataAsOf'].dropna().astype(str).unique()
    if len(asof_values) != 1:
        raise RuntimeError('News_Model_Audit.csv must contain one DataAsOf value.')
    DATA_ASOF = pd.to_datetime(asof_values[0]).strftime('%Y-%m-%d')
else:
    available = []
    for cc in ['id', 'ma', 'ph', 'th', 'vt']:
        matches = glob.glob(os.path.join(DATA_DIR, f'????-??-??_{cc}.xlsx'))
        available.append({os.path.basename(x)[:10] for x in matches})
    common_dates = set.intersection(*available) if available else set()
    if not common_dates:
        raise FileNotFoundError('No common dated ASEAN-5 input-file vintage found.')
    DATA_ASOF = max(common_dates)

# Configuration mapping countries to their exact files and prefixes.
COUNTRY_CONFIG = {
    "Indonesia": {"file": f"{DATA_ASOF}_id.xlsx", "prefix": "id_"},
    "Malaysia": {"file": f"{DATA_ASOF}_ma.xlsx", "prefix": "ma_"},
    "Philippines": {"file": f"{DATA_ASOF}_ph.xlsx", "prefix": "ph_"},
    "Thailand": {"file": f"{DATA_ASOF}_th.xlsx", "prefix": "th_"},
    "Vietnam": {"file": f"{DATA_ASOF}_vt.xlsx", "prefix": "vt_"}
}

# Translating the 3-letter codes into clean LaTeX math
TRANSFORM_MAP = {
    'pch': r"Period-to-period \%$\Delta$",
    'pca': r"\%$\Delta$ Annualized Rate",
    'chg': r"Period-to-period $\Delta$"
}

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
# 2. Helper Functions
# ==========================================
def format_date(date_val, freq):
    if pd.isna(date_val):
        return ""
    
    freq_str = str(freq).strip().lower()
    if freq_str.startswith('m'): 
        return date_val.strftime('%b %Y')
    elif freq_str.startswith('q'): 
        return f"{date_val.year} Q{date_val.quarter}"
    else:
        return date_val.strftime('%Y-%m-%d')

def get_start_end_dates(df_data, series_id, prefix, freq):
    col_in_data = series_id[len(prefix):] if series_id.lower().startswith(prefix) else series_id
    
    if col_in_data not in df_data.columns:
        return "N/A --- N/A"
    
    valid_data = df_data[['date', col_in_data]].dropna().sort_values('date')
    if valid_data.empty:
        return "N/A --- N/A"
    
    start_dt = valid_data['date'].iloc[0]
    end_dt = valid_data['date'].iloc[-1]
    
    start_str = format_date(start_dt, freq)
    end_str = format_date(end_dt, freq)
    
    return f"{start_str} --- {end_str}"

def get_clean_unit(raw_unit, series_id):
    """Parses the raw explanatory unit string into a clean academic label."""
    raw_unit_str = str(raw_unit).strip().lower()
    sid = str(series_id).lower()
    
    # Rule 3: GDP gets Percent
    if 'gdp' in sid:
        return 'Percent'
        
    # Rule 1: "Measured as level..." gets Level
    if raw_unit_str.startswith('measured as level'):
        return 'Level'
        
    # Rule 2: "Measured as index/percent..." gets sorted dynamically
    if raw_unit_str.startswith('measured as index/percent'):
        # Keywords indicating a rate or percentage
        percent_keywords = ['rate', 'ratio', 'utiliz', 'utili', 'portf', 'yield']
        if any(keyword in sid for keyword in percent_keywords):
            return 'Percent'
        else:
            return 'Index'
            
    return raw_unit # Fallback if it doesn't match the expected string patterns

def compile_latex_to_pdf(tex_snippet, base_name):
    snippet_path = os.path.join(OUT_DIR, f"{base_name}_snippet.tex")
    with open(snippet_path, "w") as f:
        f.write(tex_snippet)
        
    standalone_content = (
        "\\documentclass{article}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{longtable}\n"
        "\\usepackage[margin=1in]{geometry}\n"
        "\\begin{document}\n"
    )
    standalone_content += tex_snippet
    standalone_content += "\n\\end{document}"
    
    standalone_filename = f"{base_name}_standalone.tex"
    standalone_path = os.path.join(OUT_DIR, standalone_filename)
    
    with open(standalone_path, "w") as f:
        f.write(standalone_content)
        
    print(f"  -> Compiling PDF for {base_name}...")
    try:
        subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', standalone_filename],
            cwd=OUT_DIR, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        print("  -> Warning: pdflatex not found on system path.")

# ==========================================
# 3. Data Processing
# ==========================================
print(f"Loading specification sheet from {SPEC_FILE}...")
spec_df = pd.read_excel(SPEC_FILE)

# Apply mapping to Transformation
spec_df['Transformation_Clean'] = spec_df['Transformation'].str.strip().str.lower().map(TRANSFORM_MAP).fillna(spec_df['Transformation'])

processed_dfs = {}

for country, config in COUNTRY_CONFIG.items():
    data_path = os.path.join(DATA_DIR, config['file'])
    prefix = config['prefix']
    print(f"Processing {country} from {config['file']}...")
    
    c_spec = spec_df[spec_df['SeriesID'].str.lower().str.startswith(prefix)].copy()
    
    if c_spec.empty:
        print(f"  -> No variables found for {country}. Skipping.")
        continue
        
    df_data = pd.read_excel(data_path)
    df_data['date'] = pd.to_datetime(df_data['date'], errors='coerce')
    
    date_ranges = []
    clean_units = []
    
    for _, row in c_spec.iterrows():
        series_id = row['SeriesID']
        freq = row['Frequency']
        raw_unit = row['Units']
        
        # Apply the new dates and unit logic
        date_ranges.append(get_start_end_dates(df_data, series_id, prefix, freq))
        clean_units.append(get_clean_unit(raw_unit, series_id))
    
    c_spec['Start --- End Dates'] = date_ranges
    c_spec['Units_Clean'] = clean_units
    c_spec['Lag (days)'] = "-" 
    
    c_spec['SeriesName'] = c_spec['SeriesName'].str.split(':').str[-1].str.strip()
    
    c_table = pd.DataFrame({
        'Variable (y)': c_spec['SeriesName'],
        'Frequency': c_spec['Frequency'].str.upper(),
        'Start --- End Dates': c_spec['Start --- End Dates'],
        'Lag (days)': c_spec['Lag (days)'],
        'Unit': c_spec['Units_Clean'],
        'Transformation': c_spec['Transformation_Clean']
    })
    
    processed_dfs[country] = c_table
    
    tex_str = dataframe_to_latex(c_table, 'lllccl')
    
    full_table_str = (
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{Selected economic indicators: {country}}}\n"
        "\\resizebox{\\textwidth}{!}{\n"
        f"{tex_str}\n"
        "}\n\\end{table}"
    )
    compile_latex_to_pdf(full_table_str, f"table_{country.lower()}")

# ==========================================
# 4. Generate Combined Panel Table
# ==========================================
print("\nGenerating combined panel table...")

latex_lines = [
    r"\begin{longtable}{@{}p{0.19\textwidth}cp{0.14\textwidth}cp{0.08\textwidth}p{0.17\textwidth}@{}}",
    r"\caption{Selected economic indicators by country} \\",
    r"\toprule",
    r"\textbf{Variable (y)} & \textbf{Frequency} & \textbf{Start --- End Dates} & \textbf{Lag (days)} & \textbf{Unit} & \textbf{Transformation} \\",
    r"\midrule",
    r"\endfirsthead",
    r"\toprule",
    r"\textbf{Variable (y)} & \textbf{Frequency} & \textbf{Start --- End Dates} & \textbf{Lag (days)} & \textbf{Unit} & \textbf{Transformation} \\",
    r"\midrule",
    r"\endhead",
    r"\bottomrule",
    r"\endfoot"
]

panel_letters = ['A', 'B', 'C', 'D', 'E']
for idx, (country, df) in enumerate(processed_dfs.items()):
    
    if idx > 0:
        latex_lines.append(r"\addlinespace")
    latex_lines.append(rf"\multicolumn{{6}}{{l}}{{\textbf{{Panel {panel_letters[idx]}: {country}}}}} \\")
    latex_lines.append(r"\midrule")
    
    for _, row in df.iterrows():
        # Escape any rogue % signs in the Units column so LaTeX doesn't treat them as comments
        safe_unit = str(row['Unit']).replace('%', r'\%')
        
        row_str = f"{row['Variable (y)']} & {row['Frequency']} & {row['Start --- End Dates']} & {row['Lag (days)']} & {safe_unit} & {row['Transformation']} \\\\"
        latex_lines.append(row_str)
    
    latex_lines.append(r"\midrule")

latex_lines.extend([
    r"\end{longtable}"
])

combined_tex = "\n".join(latex_lines)
compile_latex_to_pdf(combined_tex, "table_combined_panels")

# Step 5: Clean up auxiliary compilation files
for ext in ['*.aux', '*.log']:
    for file in glob.glob(os.path.join(OUT_DIR, ext)):
        os.remove(file)

print(f"\nDone! Check the '{OUT_DIR}' directory for your updated .tex snippets and PDFs.")
