"""Generate the hard/soft appendix table without replacing validated charts.

The historical decomposition PNGs are owned by
``asean_nowcast_news_charts.m``. This legacy-named Python entry point now
only converts that MATLAB routine's audited classification CSV into the
LaTeX/PDF appendix table used by the paper.
"""

from pathlib import Path
import shutil
import subprocess

import pandas as pd


PROJECT_ROOT = Path.cwd()
CHART_DIR = (
    PROJECT_ROOT / "output" / "oos_forecasts_2000_2005"
    / "decomposition" / "charts"
)
TAXONOMY_FILE = CHART_DIR / "Hard_Soft_Indicator_Classification.csv"
AUDIT_FILE = CHART_DIR / "Decomposition_Chart_Audit.csv"

FULL_NAMES = {
    "ID": "Indonesia",
    "MA": "Malaysia",
    "PH": "Philippines",
    "TH": "Thailand",
    "VT": "Vietnam",
}


def clean_name(raw_name: str) -> str:
    value = str(raw_name).strip()
    for prefix in ("id_", "ma_", "ph_", "th_", "vt_"):
        if value.lower().startswith(prefix):
            value = value[len(prefix):]
            break
    value = value.replace("_", " ").title()
    return (
        value.replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


def validate_inputs(taxonomy: pd.DataFrame) -> None:
    required = {"Indicator", "OriginCountry", "DataType"}
    missing = required - set(taxonomy.columns)
    if missing:
        raise SystemExit(
            "The classification file is incomplete; missing columns: "
            + ", ".join(sorted(missing))
        )
    bad_types = set(taxonomy["DataType"].dropna()) - {"Hard", "Soft"}
    bad_origins = set(taxonomy["OriginCountry"].dropna()) - set(FULL_NAMES)
    if bad_types or bad_origins:
        raise SystemExit(
            f"Invalid taxonomy values. DataType={sorted(bad_types)}, "
            f"OriginCountry={sorted(bad_origins)}"
        )
    if taxonomy["Indicator"].duplicated().any():
        raise SystemExit("Indicator taxonomy contains duplicate indicators.")

    if AUDIT_FILE.exists():
        audit = pd.read_csv(AUDIT_FILE)
        valid = audit["Valid_Decomposition"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )
        residual = pd.to_numeric(audit.loc[valid, "Residual"], errors="coerce")
        max_residual = residual.abs().max()
        if pd.notna(max_residual) and max_residual > 1e-8:
            raise SystemExit(
                f"Chart audit failed: maximum valid residual={max_residual:.6g}."
            )
        print(f"Validated chart audit; maximum residual={max_residual:.6g}.")


def build_latex(taxonomy: pd.DataFrame) -> str:
    lines = [
        r"\documentclass{article}",
        r"\usepackage{booktabs}",
        r"\usepackage{tabularx}",
        r"\usepackage[margin=1in]{geometry}",
        r"\begin{document}",
        r"\section*{Appendix: Indicator Classifications by Data Type}",
    ]

    for code, full_name in FULL_NAMES.items():
        country = taxonomy[taxonomy["OriginCountry"] == code].copy()
        if country.empty:
            raise SystemExit(f"No classified indicators found for {code}.")
        country["CleanName"] = country["Indicator"].map(clean_name)
        country = country.sort_values("CleanName")
        hard = country.loc[country["DataType"] == "Hard", "CleanName"].tolist()
        soft = country.loc[country["DataType"] == "Soft", "CleanName"].tolist()
        lines.extend(
            [
                rf"\subsection*{{Classifications: {full_name}}}",
                r"\noindent\begin{tabularx}{\textwidth}{lX}",
                r"\toprule",
                r"\textbf{Classification} & \textbf{Indicators} \\",
                r"\midrule",
                rf"\textbf{{Hard}} & {', '.join(hard)} \\",
                r"\midrule",
                rf"\textbf{{Soft}} & {', '.join(soft)} \\",
                r"\bottomrule",
                r"\end{tabularx}",
                r"\vspace{1em}",
            ]
        )

    lines.append(r"\end{document}")
    return "\n".join(lines)


def main() -> None:
    if not TAXONOMY_FILE.exists():
        raise SystemExit(
            f"Missing {TAXONOMY_FILE}. Run asean_nowcast_news_charts.m first."
        )
    taxonomy = pd.read_csv(TAXONOMY_FILE)
    validate_inputs(taxonomy)
    tex_path = CHART_DIR / "hard_soft_appendix_tables.tex"
    tex_path.write_text(build_latex(taxonomy), encoding="utf-8")

    pdflatex = shutil.which("pdflatex")
    if pdflatex:
        subprocess.run(
            [pdflatex, "-interaction=nonstopmode", tex_path.name],
            cwd=CHART_DIR,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        for suffix in (".aux", ".log"):
            (CHART_DIR / f"hard_soft_appendix_tables{suffix}").unlink(
                missing_ok=True
            )
    else:
        print("Warning: pdflatex was not found; the .tex file was created.")

    print(f"Appendix taxonomy outputs saved to: {CHART_DIR}")
    print("Validated MATLAB decomposition PNGs were not modified.")


if __name__ == "__main__":
    main()
