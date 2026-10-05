import pandas as pd
import os
import subprocess
import glob

# ---------------------------------------------------------
# 1. CONFIGURATION & DICTIONARIES
# ---------------------------------------------------------
PROJECT_ROOT = os.getcwd()
LOG_FILE = os.path.join(PROJECT_ROOT, 'output', 'oos_forecasts_2000_2005', 'Variable_Selection_Log.csv')
OUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'oos_forecasts_2000_2005', 'selection_tables')

FULL_NAMES = {
    'ID': 'Indonesia', 'MA': 'Malaysia', 'PH': 'Philippines',
    'TH': 'Thailand', 'VT': 'Vietnam'
}

# The Translation Engine: Maps raw DB names to Academic labels
VAR_CLEAN_MAP = {
    # General / Cross-country
    'GDP': 'Real GDP Growth',
    'EXPORTS': 'Exports',
    'IMPORTS': 'Imports',
    'TRADE_BALANCE': 'Trade Balance',
    'TERMS_OF_TRADE': 'Terms of Trade',
    'CHINA_IMPORTS': 'Imports from China',
    'US_IMPORTS': 'Imports from United States',
    'BOP': 'Balance of Payments',
    'CURRENT_ACCOUNT': 'Current Account Balance',
    'CPI': 'Consumer Price Index (Headline)',
    'CPLHEADLINE': 'Consumer Price Index (Headline)',
    'CPI_HEADLINE': 'Consumer Price Index (Headline)',
    'CPLCORE': 'Consumer Price Index (Core)',
    'CPI_CORE': 'Consumer Price Index (Core)',
    'PPI': 'Producer Price Index',
    'FRESERVES': 'Foreign Exchange Reserves',
    'FORIEGN_RESERVES': 'Foreign Exchange Reserves',
    'F_RESERVES': 'Foreign Exchange Reserves',
    'STOCK_PRICE_INDEX': 'Stock Price Index',
    'CC': 'Consumer Confidence Index',
    'UNEMPLOYMENT_RATE': 'Unemployment Rate',
    'GLOBAL_PMI': 'Global Manufacturing PMI',

    # Indonesia
    'SURVEY_BUSINESS_ACTIVITY': 'Business Activity Survey Index',
    'DDI': 'Domestic Direct Investment',
    'FDI': 'Foreign Direct Investment',
    'IP_MANU_ML_FIRMS': 'Industrial Production: Manufacturing (Med. & Large Firms)',
    'PRIVATE_SPENDING': 'Private Consumption Spending',
    'PETROLEUM_COAL': 'Petroleum & Coal Production',
    'CAP_UTILIZ': 'Capacity Utilization Rate',
    'REAL_RETAIL_SALES': 'Real Retail Sales Index',
    'RESIDENTIAL_PROP_INDEX': 'Residential Property Price Index',
    'INVESTMENT_FINANCING': 'Investment Financing Growth',

    # Malaysia
    'BUSINESS_CONDI_INDEX': 'Business Conditions Index',
    'BUSINESS_CONDLINDEX': 'Business Conditions Index',
    'OUTPUT_PER_EMPLOYEE': 'Output per Employee',
    'IP': 'Industrial Production Index',
    'CAP_UTILI': 'Capacity Utilization Rate',
    'IP_EXCL_CONST': 'Industrial Production (Excl. Construction)',
    'IP_EXPORT_ORIENTED_INDUSTRIES': 'Industrial Production (Export-Oriented)',
    'IP_DOMESTIC_ORIENTED_INDUSTRIES': 'Industrial Production (Domestic-Oriented)',
    'MANU_SALES': 'Manufacturing Sales Value',
    'SALES_MANU_PRODUCTS': 'Sales of Manufactured Goods',
    'PROPERTY_TRANS': 'Property Transaction Volume',
    'RUBBER_PRICE': 'Rubber Price Index',
    'MA_RUBBER_PRICE': 'Rubber Price Index',
    'NUM_EMPLOY_MANU': 'Manufacturing Employment',
    'CRUDE_PALM_OIL': 'Crude Palm Oil Production',

    # Philippines
    'OUTPUT_PER_EMP': 'Output per Employee',
    'BUSI_OUTLOOK_CURR_Q': 'Business Outlook Index (Current Quarter)',
    'BUSI_OUTLOOK_NEXT_Q': 'Business Outlook Index (Next Quarter)',
    'MANU_PROD_VALUE_INDEX': 'Manufacturing Production Value Index',
    'MANU_PROD_VOLUME': 'Manufacturing Production Volume Index',
    'CC_INDEX_CURR_Q': 'Consumer Confidence (Current Quarter)',
    'CC_INDEX_NEXT_Q': 'Consumer Confidence (Next Quarter)',
    'PAST_DUE_RATIO': 'Past Due Loan Ratio',
    'DISTRESSED_ASSETS_RATIO': 'Distressed Assets Ratio',
    'RESTRUCT_LOANS_TO_LOAN_PORTF': 'Restructured Loans to Total Loan Portfolio',
    'LOAN_LOSS_TO_LOAN_PORTF': 'Loan Loss Reserves to Total Loan Portfolio',
    'AVG_PRICE_CHICKEN': 'Average Retail Price of Chicken',
    'VISTOR_ARRIVALS': 'Visitor Arrivals',
    'PRICE_EARN_RATIO': 'Price-to-Earnings Ratio',
    'M3': 'M3 Broad Money Supply',
    'BUDGET_BALANCE': 'Fiscal Budget Balance',

    # Thailand
    'NEW_MORTGAGE_LOANS_INDIV': 'New Mortgage Loans to Individuals',
    'RETAIL_SALES_VOLUME': 'Retail Sales Volume Index',
    'CONST_MATERIALS': 'Construction Materials Price Index',
    'MFG': 'Manufacturing Production Index',
    'MANU_PROD_INDEX': 'Manufacturing Production Index',
    'AG_PROD_INDEX': 'Agricultural Production Index',
    'CAR_SALES': 'Commercial & Passenger Car Sales',
    'CAPITAL_UTIL_RATE': 'Capacity Utilization Rate',
    'LAYOFF_RATE': 'Layoff Rate',
    'TOURIST_ARRIVALS': 'Foreign Tourist Arrivals',
    'NET_APPLI_SUBMI_INVE': 'Net Investment Applications Submitted (BOI)',
    'BUSISENT_3_MONTHS': 'Business Sentiment Index (3 Months Ahead)',

    # Vietnam
    'AVG_TRADING_VALUE': 'Stock Market Average Daily Trading Value',
    'VT_IMPORTS_CH': 'Imports from China',
    'IMPORTS_CH': 'Imports from China',
    'RETAILS_SALES': 'Total Retail Sales of Consumer Goods',
    'RETAIL_SALES': 'Total Retail Sales of Consumer Goods',
    'IP_EXCL_CONS': 'Industrial Production (Excl. Construction)',
    'IP_EXCL_CONST': 'Industrial Production (Excl. Construction)',
    'VEHICLE_SALES': 'Automobile & Vehicle Sales',
    'IP_TOTAL_INDUSTRY': 'Industrial Production Index (Total Industry)'
}

def clean_indicator_name(raw_name):
    s = raw_name.strip()
    
    # Check for formatting artifacts (dots instead of underscores)
    s = s.replace('.', '_')
    
    # 1. Strip the Country Prefix
    for prefix in ['ID_', 'MA_', 'PH_', 'TH_', 'VT_']:
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
            
    # 2. Check the Mapping Dictionary
    if s in VAR_CLEAN_MAP:
        return VAR_CLEAN_MAP[s]
        
    # 3. Fallback: Convert SNAKE_CASE to Title Case
    return s.replace('_', ' ').title()

def selection_frequencies(subset):
    """Return selection percentages using one row as one vintage run."""
    num_vintages = len(subset)
    counts = {}
    for row in subset['Selected_Variables'].dropna():
        # A set prevents an accidental duplicate within one log row from
        # counting the same variable twice for a single vintage.
        variables = {v.strip() for v in row.split('|') if v.strip()}
        for variable in variables:
            counts[variable] = counts.get(variable, 0) + 1
    frequencies = {
        variable: (count / num_vintages) * 100
        for variable, count in counts.items()
    } if num_vintages else {}
    return frequencies, num_vintages

def escape_latex(text):
    return text.replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')

def remove_legacy_separate_tables():
    """Remove only outputs created by the former one-method-per-table design."""
    legacy_patterns = [
        'Table_*_CORR_snippet.tex',
        'Table_*_EN_snippet.tex',
        'Table_*_CORR_standalone.tex',
        'Table_*_EN_standalone.tex',
        'Table_*_CORR_standalone.pdf',
        'Table_*_EN_standalone.pdf',
    ]
    for pattern in legacy_patterns:
        for path in glob.glob(os.path.join(OUT_DIR, pattern)):
            os.remove(path)

# ---------------------------------------------------------
# 2. GENERATION SCRIPT
# ---------------------------------------------------------
def generate_latex_tables():
    os.makedirs(OUT_DIR, exist_ok=True)
    remove_legacy_separate_tables()
    
    print("Loading Variable Selection Log...")
    try:
        df = pd.read_csv(LOG_FILE)
    except Exception as e:
        print(f"Error loading {LOG_FILE}: {e}")
        return
        
    required_methods = ('CORR', 'EN')
    for country in df['Country'].dropna().unique():
        country_data = df[df['Country'] == country]
        missing_methods = [m for m in required_methods
                           if country_data[country_data['Method'] == m].empty]
        if missing_methods:
            raise ValueError(
                f"Cannot build the combined {country} table: missing method(s) "
                + ', '.join(missing_methods)
            )

        corr_freq, corr_n = selection_frequencies(
            country_data[country_data['Method'] == 'CORR'])
        en_freq, en_n = selection_frequencies(
            country_data[country_data['Method'] == 'EN'])

        # Use the complete union so a variable important under only one rule
        # is not silently omitted. Rank by mean frequency, then maximum
        # frequency, with the raw name as a deterministic final tie-breaker.
        all_variables = set(corr_freq) | set(en_freq)
        sorted_vars = sorted(
            all_variables,
            key=lambda v: (
                -((corr_freq.get(v, 0.0) + en_freq.get(v, 0.0)) / 2),
                -max(corr_freq.get(v, 0.0), en_freq.get(v, 0.0)),
                v,
            )
        )

        full_name = FULL_NAMES.get(country, country)
        caption = f"Variable-selection frequencies: {full_name}"
        table_content = (
            f"\\begin{{table}}[htbp]\n"
            "\\centering\n"
            f"\\caption{{{caption}}}\n"
            "\\begin{tabular}{lcc}\n"
            "\\toprule\n"
            "\\textbf{Indicator} & \\textbf{Correlation (\\%)} & "
            "\\textbf{Elastic net (\\%)} \\\\\n"
            "\\midrule\n"
        )

        for variable in sorted_vars:
            clean_var = escape_latex(clean_indicator_name(variable))
            table_content += (
                f"{clean_var} & {corr_freq.get(variable, 0.0):.1f} & "
                f"{en_freq.get(variable, 0.0):.1f} \\\\\n"
            )

        table_content += (
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "\\vspace{0.35em}\n"
            "\\begin{minipage}{0.94\\linewidth}\n"
            "\\footnotesize \\textit{Notes:} Entries report the percentage of "
            "real-time vintages in which each indicator was selected. "
            f"Correlation-selection vintages: {corr_n}; elastic-net "
            f"($\\alpha=0.5$) vintages: {en_n}.\n"
            "\\end{minipage}\n"
            "\\end{table}\n"
        )

        # Save one modular snippet and one standalone document per country.
        snippet_path = os.path.join(OUT_DIR, f"Table_{country}_CORR_EN_snippet.tex")
        with open(snippet_path, 'w', encoding='utf-8') as f:
            f.write(table_content)

        standalone_content = (
            "\\documentclass{article}\n"
            "\\usepackage[margin=0.75in]{geometry}\n"
            "\\usepackage{booktabs}\n"
            "\\begin{document}\n"
            + table_content
            + "\n\\end{document}\n"
        )
        standalone_filename = f"Table_{country}_CORR_EN_standalone.tex"
        standalone_path = os.path.join(OUT_DIR, standalone_filename)
        with open(standalone_path, 'w', encoding='utf-8') as f:
            f.write(standalone_content)

        print(f"Compiling combined PDF for {country}...")
        try:
            subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', standalone_filename],
                cwd=OUT_DIR,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            print("Warning: pdflatex not found on system path.")
                
    # Step E: Clean up auxiliary compilation files
    for ext in ['*.aux', '*.log']:
        for file in glob.glob(os.path.join(OUT_DIR, ext)):
            os.remove(file)
            
    print(f"\nProcessing complete! Your clean tables and PDFs are ready in: {OUT_DIR}")

if __name__ == '__main__':
    generate_latex_tables()

