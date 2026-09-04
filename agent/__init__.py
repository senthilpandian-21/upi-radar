"""UPI Radar — multi-agent orchestration layer.

Agents:
  MonitorAgent    — 5-min prediction loop + continuous stream monitoring
  RouterAgent     — executes cascade routing, logs every decision
  AlertAgent      — SRE/merchant alerts (email/SMS) with audit trail
  QueueAgent      — graceful queuing + exponential-backoff retries
  ReportAgent     — auto-generates post-outage PDF reports
  RetrainerAgent  — weekly accuracy check + automatic retraining
"""
