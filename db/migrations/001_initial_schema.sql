-- ═══════════════════════════════════════════════════════════
-- UPI RADAR — initial schema (PostgreSQL)
-- Apply with: python db/setup.py   (or psql -f this file)
-- ═══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS bank_transactions_hourly (
    id                    SERIAL PRIMARY KEY,
    recorded_at           TIMESTAMP NOT NULL,
    bank_name             VARCHAR(50) NOT NULL,
    hour_of_day           INTEGER,
    day_of_week           INTEGER,
    day_of_month          INTEGER,
    month                 INTEGER,
    year                  INTEGER,
    is_salary_day         BOOLEAN DEFAULT FALSE,
    is_fy_end             BOOLEAN DEFAULT FALSE,
    is_month_end          BOOLEAN DEFAULT FALSE,
    is_peak_hour          BOOLEAN DEFAULT FALSE,
    is_festival_day       BOOLEAN DEFAULT FALSE,
    total_volume          BIGINT,
    success_count         BIGINT,
    failure_count         BIGINT,
    failure_rate          DECIMAL(10,6),
    technical_decline_pct DECIMAL(10,6),
    business_decline_pct  DECIMAL(10,6),
    bank_health_score     DECIMAL(5,2),
    complaint_volume      INTEGER DEFAULT 0,
    created_at            TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS razorpay_downtimes (
    id              SERIAL PRIMARY KEY,
    downtime_id     VARCHAR(50) UNIQUE,
    method          VARCHAR(50),
    bank            VARCHAR(100),
    severity        VARCHAR(20),
    status          VARCHAR(20),
    is_scheduled    BOOLEAN,
    begin_time      TIMESTAMP,
    end_time        TIMESTAMP,
    duration_mins   INTEGER,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_predictions (
    id                  SERIAL PRIMARY KEY,
    predicted_at        TIMESTAMP NOT NULL,
    prediction_horizon  INTEGER DEFAULT 120,
    outage_probability  DECIMAL(5,4),
    lstm_prob           DECIMAL(5,4),
    prophet_prob        DECIMAL(5,4),
    anomaly_score       DECIMAL(5,4),
    actual_outage       BOOLEAN DEFAULT NULL,
    prediction_correct  BOOLEAN DEFAULT NULL,
    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS routing_decisions (
    id                  SERIAL PRIMARY KEY,
    transaction_id      VARCHAR(100),
    decided_at          TIMESTAMP NOT NULL,
    original_bank       VARCHAR(50),
    routed_to_bank      VARCHAR(50),
    original_method     VARCHAR(50),
    routed_to_method    VARCHAR(50),
    reason              TEXT,
    outage_prob_at_time DECIMAL(5,4),
    bank_health_score   DECIMAL(5,2),
    success             BOOLEAN,
    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS alerts_log (
    id              SERIAL PRIMARY KEY,
    alert_type      VARCHAR(50),
    severity        VARCHAR(20),
    message         TEXT,
    sent_to         VARCHAR(200),
    outage_prob     DECIMAL(5,4),
    bank_affected   VARCHAR(100),
    method_affected VARCHAR(50),
    sent_at         TIMESTAMP DEFAULT NOW(),
    resolved_at     TIMESTAMP DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS transaction_queue (
    id              SERIAL PRIMARY KEY,
    transaction_id  VARCHAR(100) UNIQUE,
    amount          BIGINT,
    currency        VARCHAR(10) DEFAULT 'INR',
    original_method VARCHAR(50),
    original_bank   VARCHAR(50),
    queued_at       TIMESTAMP DEFAULT NOW(),
    retry_count     INTEGER DEFAULT 0,
    max_retries     INTEGER DEFAULT 5,
    status          VARCHAR(20) DEFAULT 'queued',
    processed_at    TIMESTAMP DEFAULT NULL,
    error_message   TEXT DEFAULT NULL
);

-- ── indexes on hot query paths ───────────────────────────────
CREATE INDEX IF NOT EXISTS idx_bth_recorded_at  ON bank_transactions_hourly (recorded_at);
CREATE INDEX IF NOT EXISTS idx_bth_bank_time    ON bank_transactions_hourly (bank_name, recorded_at);
CREATE INDEX IF NOT EXISTS idx_mp_predicted_at  ON model_predictions (predicted_at DESC);
CREATE INDEX IF NOT EXISTS idx_rd_decided_at    ON routing_decisions (decided_at DESC);
CREATE INDEX IF NOT EXISTS idx_al_sent_at       ON alerts_log (sent_at DESC);
CREATE INDEX IF NOT EXISTS idx_tq_status        ON transaction_queue (status);
