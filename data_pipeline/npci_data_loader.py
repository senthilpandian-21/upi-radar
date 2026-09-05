"""NPCI data loader — reads NPCI monthly UPI statistics CSVs.

Expected columns (loosely matched): month, bank_name / remitter_bank,
total_volume, success_count, failure_count, technical_decline_pct,
business_decline_pct. Robust to whatever real header layout NPCI ships,
and ships a bundled fallback dataset so the pipeline works offline.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
for _ in range(4):
    if (ROOT / "config.py").exists():
        break
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from io import StringIO

import pandas as pd
from loguru import logger

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Bundled sample of NPCI-style monthly statistics (bank-wise TD %).
FALLBACK_CSV = """month,bank_name,total_volume,success_count,failure_count,technical_decline_pct,business_decline_pct
2025-04,SBI,2581000000,2569000000,12000000,0.0092,0.0031
2025-04,HDFC,2124000000,2122000000,2000000,0.0012,0.0009
2025-04,ICICI,1843000000,1837000000,6000000,0.0103,0.0022
2025-04,AXIS,1291000000,1285000000,6000000,0.0061,0.0017
2025-04,PNB,811000000,801000000,10000000,0.0122,0.0040
2025-04,BOB,622000000,613000000,9000000,0.0151,0.0035
2025-04,KOTAK,875000000,872000000,3000000,0.0041,0.0011
2025-04,YES,310000000,309000000,1000000,0.0030,0.0012
2025-05,SBI,2620000000,2609000000,11000000,0.0089,0.0030
2025-05,HDFC,2150000000,2148000000,2000000,0.0013,0.0008
2025-05,ICICI,1865000000,1860000000,5000000,0.0101,0.0021
2025-05,AXIS,1310000000,1304000000,6000000,0.0060,0.0016
2025-05,PNB,825000000,815000000,10000000,0.0120,0.0039
2025-05,BOB,630000000,621000000,9000000,0.0150,0.0034
2025-05,KOTAK,890000000,887000000,3000000,0.0040,0.0010
2025-05,YES,315000000,314000000,1000000,0.0031,0.0011
"""


class NPCIDataLoader:
    """Load + clean NPCI monthly statistics into a tidy dataframe."""

    def __init__(self, raw_dir: Path | None = None):
        self.raw_dir = Path(raw_dir) if raw_dir else RAW_DATA_DIR

    def list_files(self) -> list[Path]:
        return sorted(self.raw_dir.glob("*.csv"))

    def load(self, use_fallback: bool = True) -> pd.DataFrame:
        files = self.list_files()
        frames = []
        for path in files:
            try:
                frame = self._load_one(path)
                if frame is not None and not frame.empty:
                    frames.append(frame)
            except Exception as exc:
                logger.warning(f"Skipping {path.name}: {exc}")

        if not frames:
            if use_fallback:
                logger.info("No NPCI CSV found — loading bundled fallback dataset")
                frames = [pd.read_csv(StringIO(FALLBACK_CSV))]
            else:
                return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        return self.clean(df)

    @staticmethod
    def _load_one(path: Path) -> pd.DataFrame:
        return pd.read_csv(path)

    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        """Normalise column names & types to a stable schema."""
        df = df.copy()
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

        rename = {}
        for col in df.columns:
            if "remitter_bank" in col or "bank_name" in col or "bank" == col:
                rename[col] = "bank_name"
            elif "technical" in col:
                rename[col] = "technical_decline_pct"
            elif "business" in col:
                rename[col] = "business_decline_pct"
        df = df.rename(columns=rename)

        for col in ("total_volume", "success_count", "failure_count"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
        for col in ("technical_decline_pct", "business_decline_pct"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        if "month" in df.columns:
            df["month"] = pd.to_datetime(df["month"].astype(str), errors="coerce")
        return df.dropna(subset=["bank_name"])

    def bank_base_td_rates(self, df: pd.DataFrame | None = None) -> dict:
        """Map bank → mean technical decline rate (used as model feature)."""
        if df is None:
            df = self.load()
        if "technical_decline_pct" not in df.columns:
            return {}
        grouped = (df.groupby("bank_name")["technical_decline_pct"].mean().to_dict())
        return {k.upper(): float(v) for k, v in grouped.items()}

    def save_processed(self, df: pd.DataFrame, output: str = "data/processed/npci_clean.csv") -> None:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info(f"Saved {len(df)} rows → {out}")
        return out


if __name__ == "__main__":  # pragma: no cover
    loader = NPCIDataLoader()
    data = loader.load()
    print(f"Loaded {len(data)} rows")
    print(data.head(10).to_string())
