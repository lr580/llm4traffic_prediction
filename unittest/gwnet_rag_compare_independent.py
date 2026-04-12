import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "utils" / "baselines" / "gwnet_rag.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "unittest" / "gwnet_rag_compare_table.xlsx"
DATASET_ORDER = ["PEMS03", "PEMS04", "PEMS07", "PEMS08", "SD", "GBA", "GLA", "CA"]
METRICS = ["mae", "mape", "rmse"]
SORT_DATASET = "SD"
SORT_METRIC = "mae"


def _float_or_none(text: str):
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def resolve_datasets(rows):
    found = {row.get("dataset", "").strip() for row in rows if row.get("dataset")}
    ordered = [d for d in DATASET_ORDER if d in found]
    remaining = sorted(found - set(ordered))
    return ordered + remaining


def build_table(rows, datasets):
    models = sorted({row.get("model", "").strip() for row in rows if row.get("model")})
    values = {}
    for row in rows:
        model = row.get("model", "").strip()
        dataset = row.get("dataset", "").strip()
        if not model or not dataset:
            continue
        for metric in METRICS:
            values[(model, dataset, metric)] = row.get(metric, "").strip()

    def sort_key(model):
        raw = values.get((model, SORT_DATASET, SORT_METRIC), "")
        score = _float_or_none(raw)
        if score is None:
            return (1, float("inf"), model)
        return (0, score, model)

    sorted_models = sorted(models, key=sort_key)
    table = []
    for model in sorted_models:
        row = [model]
        for dataset in datasets:
            for metric in METRICS:
                row.append(values.get((model, dataset, metric), ""))
        table.append(row)
    return table


def build_header_rows(datasets):
    level_1 = ["model"]
    level_2 = [""]
    for dataset in datasets:
        level_1.extend([dataset, "", ""])
        level_2.extend(METRICS)
    return level_1, level_2


def write_table_csv(output_path: Path, datasets, table):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        h1, h2 = build_header_rows(datasets)
        writer.writerow(h1)
        writer.writerow(h2)
        writer.writerows(table)


def write_table_xlsx(output_path: Path, datasets, table):
    try:
        from openpyxl import Workbook
    except ImportError as exc:
        raise RuntimeError("Excel export requires openpyxl. Install it or use --output *.csv.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "gwnet_compare"

    ws.cell(row=1, column=1, value="model")
    ws.merge_cells(start_row=1, start_column=1, end_row=2, end_column=1)
    for i, dataset in enumerate(datasets):
        col_start = 2 + i * len(METRICS)
        col_end = col_start + len(METRICS) - 1
        ws.cell(row=1, column=col_start, value=dataset)
        ws.merge_cells(start_row=1, start_column=col_start, end_row=1, end_column=col_end)
        for j, metric in enumerate(METRICS):
            ws.cell(row=2, column=col_start + j, value=metric)

    for r, row_data in enumerate(table, start=3):
        for c, value in enumerate(row_data, start=1):
            ws.cell(row=r, column=c, value=value)

    ws.freeze_panes = "B3"
    wb.save(output_path)


def write_table(output_path: Path, datasets, table):
    ext = output_path.suffix.lower()
    if ext == ".xlsx":
        write_table_xlsx(output_path, datasets, table)
    elif ext == ".csv":
        write_table_csv(output_path, datasets, table)
    else:
        raise ValueError("Unsupported output extension. Use .csv or .xlsx")


def main():
    parser = argparse.ArgumentParser(
        description="Build model-vs-dataset metric comparison table from gwnet_rag.csv."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input CSV path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path.")
    args = parser.parse_args()

    rows = read_rows(args.input)
    datasets = resolve_datasets(rows)
    table = build_table(rows, datasets)
    write_table(args.output, datasets, table)

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Rows:   {len(table)} models")
    print(f"Cols:   {len(datasets) * len(METRICS)} metrics columns ({len(datasets)} datasets x {len(METRICS)} metrics)")
    print("Header: two-level (dataset -> mae/mape/rmse)")
    print(f"Sorted by: {SORT_DATASET}_{SORT_METRIC} ascending (missing values at bottom)")


if __name__ == "__main__":
    main()
