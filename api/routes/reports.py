"""Report endpoints — list / download / generate PDF outage reports."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from agent.report_agent import ReportAgent
from config import reports_dir

router = APIRouter()


@router.get("/list", summary="All generated PDF reports")
async def list_reports() -> dict:
    reports = ReportAgent().list_reports()
    return {"count": len(reports), "reports": reports}


@router.get("/download/{filename}", summary="Download a report PDF")
async def download_report(filename: str) -> FileResponse:
    # hard traversal guard: basename only + resolved-path containment
    safe = Path(filename).name
    base = reports_dir().resolve()
    path = (base / safe).resolve()
    if (not path.is_relative_to(base) or not path.exists()
            or path.suffix.lower() != ".pdf"):
        raise HTTPException(status_code=404, detail=f"Report '{filename}' not found")
    return FileResponse(str(path), media_type="application/pdf",
                        filename=safe)


@router.post("/generate", summary="Generate a demo post-outage report")
async def generate_report(hours_ago: float = 2.0) -> dict:
    """Builds a PDF from the recent routing audit trail (demo-friendly)."""
    end = datetime.now()
    start = end - timedelta(hours=hours_ago)
    try:
        path = ReportAgent().generate_post_outage_report(start, end)
        return {"status": "ok", "file": path, "filename": path.split("/")[-1]}
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
