"""UPI Radar — deterministic policy engine.

Rules are intentionally hard-coded & auditable. AI decides *what is
likely to break*; this layer decides *what to do* — and compliance-
relevant routing decisions must never be made by a black box.
"""
