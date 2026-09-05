"""Alert agent — pages the SRE team & merchants (email/SMS).

Every alert is appended to audit_logs/alerts_sent.log as JSON and, when
the DB is available, to the alerts_log table. Delivery failures are
logged — never raised — so alerting can never take the system down.
"""
from __future__ import annotations

import json
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from loguru import logger

from config import audit_path, is_demo_mode

try:
    from twilio.rest import Client as TwilioClient

    TWILIO_AVAILABLE = True
except Exception:  # pragma: no cover
    TwilioClient = None
    TWILIO_AVAILABLE = False



class AlertAgent:
    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("ALERT_EMAIL_FROM", "")
        self.sre_email = os.getenv("ALERT_EMAIL_TO", "sre-team@razorpay.com")
        self.demo_mode = is_demo_mode()
        self._twilio = None

    # ── SRE alert ───────────────────────────────────────────────────
    async def send_sre_alert(self, probability: float,
                             reason: str, prediction_details: dict | None = None) -> bool:
        """High/medium risk → page SRE with model breakdown + action."""
        prediction_details = prediction_details or {}
        risk_emoji = "🔴" if probability > 0.85 else "🟡"
        subject = (f"{risk_emoji} UPI Radar: Outage Risk {probability:.0%} "
                   f"— Action Required")
        body = self._compose_sre_body(probability, reason, prediction_details)

        sent = self._send_email(to=self.sre_email, subject=subject, body=body)
        self._log_alert("SRE", probability, reason, sent_to=self.sre_email,
                        severity="HIGH" if probability > 0.85 else "MEDIUM")
        self._send_sms(f"UPI Radar: outage risk {probability:.0%}. {reason}")
        return sent

    def _compose_sre_body(self, probability: float, reason: str,
                          details: dict) -> str:
        recommended = ("IMMEDIATE: scale up API servers + check bank "
                       "integrations (UPI → fallback rails ready)")
        if probability <= 0.85:
            recommended = ("Monitor closely. Consider pre-scaling API "
                           "servers and enabling bank fallbacks.")
        return f"""UPI RADAR — SRE ALERT
{'=' * 50}

Outage Probability: {probability:.1%}
Risk Level: {details.get('risk_level', '—')}
Detected At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Horizon: {details.get('horizon_minutes', 120)} minutes

REASON
{reason}

MODEL BREAKDOWN
• LSTM Model:      {details.get('lstm_probability', 0):.1%}
• Prophet Model:   {details.get('prophet_probability', 0):.1%}
• Anomaly Score:   {details.get('anomaly_probability', 0):.1%}
• Models live:     {details.get('models_loaded', {})}

RECOMMENDED ACTION
{recommended}

Dashboard: http://localhost:8501
— UPI Radar Automated System
"""

    # ── merchant alert ──────────────────────────────────────────────
    async def send_merchant_alert(self, transaction_id: str,
                                  bank_decision=None,
                                  gateway_decision=None) -> None:
        """Routing-change notice (merchant dashboard/webhook feed)."""
        message = "Payment rerouted"
        if bank_decision is not None and getattr(bank_decision, "reason", ""):
            message += f": {bank_decision.reason}"
        elif gateway_decision is not None and getattr(gateway_decision, "reason", ""):
            message += f": {gateway_decision.reason}"
        data = {"type": "ROUTING_CHANGE", "transaction_id": transaction_id,
                "message": message, "timestamp": datetime.now().isoformat()}
        self._log_alert("MERCHANT", 0.0, json.dumps(data),
                        sent_to="merchant-dashboard")
        logger.info(f"Merchant alert: {message}")

    # ── delivery ────────────────────────────────────────────────────
    def _send_email(self, to: str, subject: str, body: str) -> bool:
        """SMTP send; demo-safe when no real credentials configured."""
        placeholder = (not self.from_email or not self.smtp_password
                       or "your_app_password" in self.smtp_password
                       or "XXX" in self.from_email)
        if self.demo_mode or placeholder:
            logger.info(f"[demo] Email queued → {to}: {subject}")
            return True
        try:
            msg = MIMEMultipart()
            msg["From"] = self.from_email
            msg["To"] = to
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as server:
                server.starttls()
                server.login(self.from_email, self.smtp_password)
                server.send_message(msg)
            logger.success(f"Email sent → {to}")
            return True
        except Exception as exc:
            logger.warning(f"Email send failed ({exc}) — alert still logged")
            return False

    def _send_sms(self, message: str, to_number: str | None = None) -> bool:
        """Optional Twilio SMS (never blocks the pipeline)."""
        if not TWILIO_AVAILABLE:
            return False
        sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        token = os.getenv("TWILIO_AUTH_TOKEN", "")
        from_number = os.getenv("TWILIO_FROM_NUMBER", "")
        if not all([sid, token, from_number]) or "XXX" in sid:
            return False
        try:
            if self._twilio is None:
                self._twilio = TwilioClient(sid, token)
            self._twilio.messages.create(
                body=message[:1600], from_=from_number,
                to=to_number or os.getenv("TWILIO_TO_NUMBER", ""))
            return True
        except Exception as exc:
            logger.debug(f"SMS failed ({exc})")
            return False

    # ── audit ───────────────────────────────────────────────────────
    def _log_alert(self, alert_type: str, probability: float, reason: str,
                   sent_to: str = "", severity: str = "MEDIUM",
                   bank_affected: str = "", method_affected: str = "") -> None:
        entry = {"timestamp": datetime.now().isoformat(), "type": alert_type,
                 "severity": severity, "probability": round(probability, 4),
                 "reason": reason, "sent_to": sent_to,
                 "bank_affected": bank_affected, "method_affected": method_affected}
        try:
            with audit_path("alerts_sent.log").open("a") as handle:
                handle.write(json.dumps(entry) + "\n")
        except OSError:
            pass
        try:
            from db.models import AlertLog, session_scope

            with session_scope() as session:
                session.add(AlertLog(
                    alert_type=alert_type, severity=severity,
                    message=str(reason)[:2000], sent_to=sent_to[:200],
                    outage_prob=round(probability, 4),
                    bank_affected=bank_affected or None,
                    method_affected=method_affected or None))
        except Exception as exc:
            logger.debug(f"Alert DB log skipped: {exc}")
