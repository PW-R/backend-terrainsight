from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"
ENCODINGS = ["utf-8-sig", "utf-8", "cp874", "tis-620"]


def read_csv(path):
    errors = []

    for encoding in ENCODINGS:
        try:
            df = pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False,
            )
            return df, encoding

        except UnicodeDecodeError:
            errors.append(f"{encoding}: decode failed")

    raise ValueError("; ".join(errors))


def main():
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์: {DATA_DIR}")

    files = sorted(
        path
        for path in DATA_DIR.rglob("*")
        if path.is_file()
    )

    csv_files = [
        path for path in files
        if path.suffix.lower() == ".csv"
    ]

    print(f"DATA DIRECTORY: {DATA_DIR}")
    print(f"TOTAL FILES: {len(files)}")
    print(f"CSV FILES: {len(csv_files)}")

    print("\nFILE LIST")
    for index, path in enumerate(files, start=1):
        print(f"{index}. {path.relative_to(DATA_DIR)}")

    successful = 0
    failures = []

    for index, path in enumerate(csv_files, start=1):
        print("\n" + "=" * 80)
        print(f"DATASET {index}/{len(csv_files)}")
        print(f"FILE: {path.relative_to(DATA_DIR)}")
        print(f"DATASET KEY: {path.stem}")
        print("=" * 80)

        try:
            df, encoding = read_csv(path)

            print(f"ENCODING: {encoding}")
            print(f"SHAPE: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"DUPLICATE ROWS: {int(df.duplicated().sum())}")

            # repr ทำให้เห็นช่องว่างหรืออักขระแฝงในชื่อคอลัมน์
            print("\nEXACT COLUMN NAMES")
            for number, col in enumerate(df.columns, start=1):
                print(f"{number}. {col!r}")

            schema = pd.DataFrame({
                "column": [repr(col) for col in df.columns],
                "dtype": [str(dtype) for dtype in df.dtypes],
                "missing": df.isna().sum().to_numpy(),
                "unique": df.nunique(dropna=True).to_numpy(),
            })

            print("\nCOLUMN SUMMARY")
            print(schema.to_string(index=False))

            print("\nFIRST 3 ROWS")
            if df.empty:
                print("(empty dataset)")
            else:
                print(df.head(3).to_string(index=False))

            successful += 1

        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print(f"FAILED: {message}")
            failures.append((str(path.relative_to(DATA_DIR)), message))

    print("\n" + "=" * 80)
    print("SCAN SUMMARY")
    print(f"Successfully loaded: {successful}/{len(csv_files)}")
    print(f"Failed: {len(failures)}")

    for filename, error in failures:
        print(f"- {filename}: {error}")

    if not csv_files:
        print("ไม่พบไฟล์ CSV ภายในโฟลเดอร์ data")


if __name__ == "__main__":
    main()