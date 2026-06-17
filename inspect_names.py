import pandas as pd

def get_projects(path):
    xls = pd.ExcelFile(path)
    projects = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        df.columns = df.columns.astype(str).str.strip().str.replace("*", "", regex=False)
        if "Project Name" in df.columns:
            projs = df["Project Name"].dropna().unique().tolist()
            projects.extend(projs)
    return sorted(list(set(projects)))

p25 = get_projects('data/evidence/data_procesor/qualityReportIDBTest_December_2025.xlsx')
p26 = get_projects('data/evidence/data_procesor/qualityReportIDBTest_May_2026.xlsx')

print("2025 Projects:")
for p in p25: print(" -", repr(p))
print("\n2026 Projects:")
for p in p26: print(" -", repr(p))
