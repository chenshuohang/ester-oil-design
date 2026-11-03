from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SMILES_COL = "SMILES"
ID_COL = "Name"
TARGET_COL = "Flash_Point"


def _load_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError as exc:
        raise ImportError(
            "RDKit is required. Run this workflow with "
            "`qsar/.venv_smiles/bin/python`."
        ) from exc
    return Chem, Descriptors


def read_table(path: str | Path, sheet_name: str | int | None = 0) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".xlsx":
        try:
            return pd.read_excel(path, sheet_name=sheet_name)
        except ImportError:
            return _read_xlsx_without_openpyxl(path, sheet_name)
    if suffix == ".xls":
        return pd.read_excel(path, sheet_name=sheet_name)
    raise ValueError(f"Unsupported table format: {path.suffix}")


def load_smiles(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".smi", ".smiles", ".txt"}:
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle, start=1):
                parts = line.strip().split()
                if not parts:
                    continue
                smiles = parts[0]
                mol_id = parts[1] if len(parts) > 1 else f"mol_{idx:04d}"
                records.append({SMILES_COL: smiles, ID_COL: mol_id})
        return pd.DataFrame(records, columns=[SMILES_COL, ID_COL])

    df = read_table(path)
    if SMILES_COL not in df.columns:
        if len(df.columns) == 1:
            df = df.rename(columns={df.columns[0]: SMILES_COL})
        else:
            raise ValueError(f"Cannot find a '{SMILES_COL}' column in {path}")

    if ID_COL not in df.columns:
        df[ID_COL] = [f"mol_{i + 1:04d}" for i in range(len(df))]

    df = df[[SMILES_COL, ID_COL]].copy()
    df[SMILES_COL] = df[SMILES_COL].astype(str).str.strip()
    df = df[~df[SMILES_COL].isin(["", "nan", "NaN", "None"])]
    return df.reset_index(drop=True)


def calculate_rdkit_descriptors(
    smiles: Iterable[str], names: Iterable[str] | None = None
) -> pd.DataFrame:
    Chem, Descriptors = _load_rdkit()
    smiles_list = list(smiles)
    name_list = list(names) if names is not None else [
        f"mol_{i + 1:04d}" for i in range(len(smiles_list))
    ]
    if len(smiles_list) != len(name_list):
        raise ValueError("smiles and names must have the same length")

    descriptor_functions = Descriptors._descList
    rows = []
    for smi, name in zip(smiles_list, name_list):
        mol = Chem.MolFromSmiles(str(smi))
        row = {ID_COL: name, SMILES_COL: smi, "valid": mol is not None}
        if mol is None:
            row.update({desc_name: np.nan for desc_name, _ in descriptor_functions})
        else:
            for desc_name, desc_func in descriptor_functions:
                try:
                    row[desc_name] = desc_func(mol)
                except Exception:
                    row[desc_name] = np.nan
        rows.append(row)

    columns = [ID_COL, SMILES_COL, "valid", *[name for name, _ in descriptor_functions]]
    return pd.DataFrame(rows, columns=columns)


def rdkit_descriptors_from_file(
    input_path: str | Path, output_path: str | Path | None = None
) -> pd.DataFrame:
    df = load_smiles(input_path)
    desc_df = calculate_rdkit_descriptors(df[SMILES_COL], df[ID_COL])
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        desc_df.to_csv(output_path, index=False)
    return desc_df


def pick_first_available(df: pd.DataFrame, candidates: list[str], role: str) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    lower_map = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    raise ValueError(f"Cannot find {role} column. Available columns: {list(df.columns)}")


def extract_numeric(value) -> float:
    if pd.isna(value):
        return np.nan
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value).replace("＜", "<"))
    if not match:
        return np.nan
    return float(match.group(0))


def _read_xlsx_without_openpyxl(path: Path, sheet_name: str | int | None) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        shared_strings = _read_shared_strings(archive)
        sheet_path = _resolve_sheet_path(archive, sheet_name)
        root = ET.fromstring(archive.read(sheet_path))

    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows = []
    for row_node in root.findall(".//x:sheetData/x:row", namespace):
        values = {}
        for cell in row_node.findall("x:c", namespace):
            ref = cell.attrib.get("r", "")
            col_idx = _column_index(ref)
            values[col_idx] = _cell_value(cell, shared_strings, namespace)
        if values:
            rows.append([values.get(i) for i in range(max(values) + 1)])

    if not rows:
        return pd.DataFrame()
    width = max(len(row) for row in rows)
    rows = [row + [None] * (width - len(row)) for row in rows]
    header = [_clean_header(value) for value in rows[0]]
    return pd.DataFrame(rows[1:], columns=header)


def _read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    strings = []
    for si in root.findall("x:si", namespace):
        parts = [node.text or "" for node in si.findall(".//x:t", namespace)]
        strings.append("".join(parts))
    return strings


def _resolve_sheet_path(archive: zipfile.ZipFile, sheet_name: str | int | None) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    workbook_ns = {
        "x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    rel_ns = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}

    sheets = workbook.findall(".//x:sheet", workbook_ns)
    if sheet_name is None:
        sheet_idx = 0
    elif isinstance(sheet_name, int):
        sheet_idx = sheet_name
    else:
        names = [sheet.attrib.get("name") for sheet in sheets]
        if sheet_name not in names:
            raise ValueError(f"Sheet '{sheet_name}' not found. Available sheets: {names}")
        sheet_idx = names.index(sheet_name)

    sheet = sheets[sheet_idx]
    rel_id = sheet.attrib[f"{{{workbook_ns['r']}}}id"]
    rel_targets = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall("r:Relationship", rel_ns)
    }
    target = rel_targets[rel_id]
    return target if target.startswith("xl/") else f"xl/{target}"


def _cell_value(cell: ET.Element, shared_strings: list[str], namespace: dict[str, str]):
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//x:t", namespace))

    value_node = cell.find("x:v", namespace)
    if value_node is None or value_node.text is None:
        return None
    raw = value_node.text
    if cell_type == "s":
        return shared_strings[int(raw)]
    if cell_type == "b":
        return bool(int(raw))
    return _coerce_number(raw)


def _coerce_number(raw: str):
    if re.fullmatch(r"[-+]?\d+", raw):
        return int(raw)
    if re.fullmatch(r"[-+]?\d*\.\d+(e[-+]?\d+)?", raw, flags=re.IGNORECASE):
        return float(raw)
    return raw


def _column_index(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref)
    if not letters:
        return 0
    index = 0
    for char in letters.group(0):
        index = index * 26 + ord(char) - ord("A") + 1
    return index - 1


def _clean_header(value) -> str:
    if value is None:
        return ""
    return str(value).strip()
