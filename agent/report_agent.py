"""Report agent — auto-generates post-outage PDF reports.

Called by the monitor when an outage resolves (production) or manually
via POST /reports/generate (demo). ReportLab is imported lazily so the
rest of the system never depends on it being installed.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from loguru import logger

import sys

ROOT = Path(__file__).resolve()
for _ in range(4):
    if (ROOT / "config.py").exists():
        break
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from config import audit_path, reports_dir

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer,
                                    Table, TableStyle)

    REPORTLAB_AVAILABLE = True
except Exception:  # pragma: no cover
    REPORTLAB_AVAILABLE = False


class ReportAgent:
    @staticmethod
    def _outage_payload_from_log(limit: int = 200) -> list[dict]:
        """Rebuild a routing-decision story from the JSON audit log."""
        path = audit_path("routing_decisions.log")
        if not path.exists():
            return []
        lines = [line for line in path.read_text().splitlines()
                 if line.strip()][-limit:]
        return [json.loads(line) for line in lines]

    # ── post-outage report ──────────────────────────────────────────
    def generate_post_outage_report(self,
                                    outage_start: datetime,
                                    outage_end: datetime,
                                    affected_banks: list[str] | None = None,
                                    routing_decisions: list[dict] | None = None
                                    ) -> str:
        """
        Build the PDF. Returns the file path.
        """
        if not REPORTLAB_AVAILABLE:
            raise ImportError("reportlab not installed — "
                              "pip install -r requirements.txt")
        affected_banks = affected_banks or ["SBI", "BOB", "PNB"]
        routing_decisions = (routing_decisions
                             if routing_decisions is not None
                             else self._outage_payload_from_log())

        filename = (f"outage_report_"
                    f"{outage_start.strftime('%Y%m%d_%H%M')}.pdf")
        out_path = reports_dir() / filename

        duration_min = max(0, int((outage_end - outage_start).total_seconds() // 60))
        total_txns = len(routing_decisions)
        successful = sum(1 for r in routing_decisions if r.get("success"))
        routed = sum(1 for r in routing_decisions
                     if r.get("routed_to_bank")
                     and r["routed_to_bank"] != r.get("original_bank"))
        success_rate = (successful / total_txns) if total_txns else 0.0

        doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                                title=f"UPI Radar — Post-Outage Report {filename}")
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("TitleX", parent=styles["Title"],
                                     textColor=colors.HexColor("#0a2b45"))
        h2 = ParagraphStyle("H2X", parent=styles["Heading2"],
                            textColor=colors.HexColor("#123c5f"))

        story = []
        story.append(Paragraph("⚡ UPI RADAR — POST-OUTAGE REPORT", title_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                               styles["Normal"]))
        story.append(Spacer(1, 0.15 * inch))

        summary = [
            ["Outage start", outage_start.strftime("%Y-%m-%d %H:%M:%S")],
            ["Outage end", outage_end.strftime("%Y-%m-%d %H:%M:%S")],
            ["Duration", f"{duration_min} minutes"],
            ["Banks affected", ", ".join(affected_banks)],
            ["Transactions processed", str(total_txns)],
            ["Transactions rerouted", str(routed)],
            ["Routing success rate", f"{success_rate:.1%}"],
        ]
        table = Table(summary, colWidths=[2.2 * inch, 4.3 * inch])
        table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef3f8")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b7c7d7")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.2 * inch))

        # ── routing decision detail ──────────────────────────────────
        story.append(Paragraph("Routing decisions (audit trail)", h2))
        rows = [["#", "Time", "Original", "Routed to", "Method", "Success"]]
        for i, decision in enumerate(routing_decisions[-15:], start=1):
            rows.append([
                str(i),
                str(decision.get("timestamp", ""))[11:19],
                str(decision.get("original_bank", "?")),
                str(decision.get("routed_to_bank") or decision.get("original_bank", "-")),
                str(decision.get("routed_to_method") or decision.get("original_method", "-")),
                "✅" if decision.get("success") else "❌",
            ])
        detail = Table(rows, colWidths=[0.5 * inch, 1.0 * inch, 1.1 * inch,
                                        1.2 * inch, 0.9 * inch, 0.8 * inch])
        detail.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0a2b45")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c5d2de")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f4f7fa")]),
        ]))
        story.append(detail)
        story.append(Spacer(1, 0.2 * inch))

        # ── recommendations ──────────────────────────────────────────
        story.append(Paragraph("Recommendations", h2))
        recs = [
            "• Scale up API server capacity ahead of the next salary-day window.",
            "• Review PSP/bank integration timeouts for affected banks.",
            "• Keep smart routing enabled: it protected "
            f"{success_rate:.0%} of affected transactions.",
            "• File RCA with the affected bank using the audit trail above.",
        ]
        for rec in recs:
            story.append(Paragraph(rec, styles["Normal"]))
            story.append(Spacer(1, 0.05 * inch))

        doc.build(story)
        logger.success(f"Post-outage report generated → {out_path}")
        return str(out_path)

    # ── daily report ────────────────────────────────────────────────
    def generate_daily_report(self, day: datetime | None = None) -> str:
        day = day or datetime.now()
        start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start.replace(hour=23, minute=59, second=59)
        decisions = [d for d in self._outage_payload_from_log(limit=1000)]
        if decisions:
            try:
                decisions = [d for d in decisions
                             if start.strftime("%Y-%m-%d")
                             in str(d.get("timestamp", ""))]
            except Exception:
                pass
        return self.generate_post_outage_report(start, end,
                                                 affected_banks=["—"],
                                                 routing_decisions=decisions)

    # ── listing ─────────────────────────────────────────────────────
    def list_reports(self) -> list[dict]:
        out = []
        for path in sorted(reports_dir().glob("*.pdf")):
            out.append({"filename": path.name, "size_bytes": path.stat().st_size,
                        "created": datetime.fromtimestamp(path.stat().st_mtime)
                        .strftime("%Y-%m-%d %H:%M:%S"),
                        "path": str(path)})
        return out


if __name__ == "__main__":  # pragma: no cover
    agent = ReportAgent()
    now = datetime.now()
    path = agent.generate_post_outage_report(now, now)
    print("Report:", path)
    print(agent.list_reports())
