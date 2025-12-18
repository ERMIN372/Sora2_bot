BEGIN;

CREATE TABLE IF NOT EXISTS users (
    user_id BIGINT PRIMARY KEY,
    credits INTEGER NOT NULL DEFAULT 0,
    bonus_granted BOOLEAN NOT NULL DEFAULT FALSE,
    economy_v2 BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    notes TEXT,
    username TEXT,
    first_name TEXT,
    last_name TEXT,
    display_name TEXT,
    tg_link TEXT
);

COMMENT ON TABLE users IS 'Telegram users with credit balances';
COMMENT ON COLUMN users.economy_v2 IS 'Marks users migrated to the new credit economy';
COMMENT ON COLUMN users.bonus_granted IS 'Prevents repeated welcome bonuses';

CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users (created_at);

CREATE TABLE IF NOT EXISTS payments (
    id BIGSERIAL PRIMARY KEY,
    provider TEXT NOT NULL,
    ext_id TEXT NOT NULL,
    user_id BIGINT REFERENCES users(user_id) ON DELETE SET NULL,
    amount_cp INTEGER NOT NULL,
    items INTEGER,
    price_rub NUMERIC(12, 2),
    credits_bought INTEGER,
    bonus_credits INTEGER,
    bonus_pct NUMERIC(6, 4),
    status TEXT NOT NULL,
    payload TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    idempotency_key TEXT,
    username TEXT,
    package_id TEXT,
    purchased_credits INTEGER,
    processed_at TIMESTAMPTZ,
    type TEXT,
    ref_payment_id TEXT
);

COMMENT ON TABLE payments IS 'Incoming payments (Stars, YooKassa, manual adjustments)';
COMMENT ON COLUMN payments.ext_id IS 'Provider payment identifier (charge_id/order_id)';
COMMENT ON COLUMN payments.idempotency_key IS 'Client-provided idempotency key to deduplicate retries';

CREATE UNIQUE INDEX IF NOT EXISTS idx_payments_provider_ext ON payments (provider, ext_id);
CREATE INDEX IF NOT EXISTS idx_payments_status ON payments (status);
CREATE INDEX IF NOT EXISTS idx_payments_user ON payments (user_id);
CREATE INDEX IF NOT EXISTS idx_payments_idempotency ON payments (idempotency_key) WHERE idempotency_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    user_id BIGINT REFERENCES users(user_id) ON DELETE SET NULL,
    prompt TEXT NOT NULL,
    image_file_id TEXT,
    sora_req_id TEXT,
    status TEXT NOT NULL,
    video_url TEXT,
    video_id TEXT,
    file_url TEXT,
    operation_name TEXT,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    size TEXT,
    seconds INTEGER,
    model TEXT,
    cost_credits INTEGER,
    username TEXT,
    corr_id TEXT,
    idempotency_key TEXT,
    content_type TEXT NOT NULL DEFAULT 'video',
    metadata JSONB,
    status_message_id INTEGER,
    status_message_index INTEGER NOT NULL DEFAULT 0,
    status_message_updated_at TIMESTAMPTZ
);

COMMENT ON TABLE jobs IS 'Generation jobs across Gemini/Sora providers';
COMMENT ON COLUMN jobs.content_type IS 'Type of generated asset (video/image)';
COMMENT ON COLUMN jobs.metadata IS 'Provider-specific metadata stored as JSON';

CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs (status, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_user_status ON jobs (user_id, status);
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_idempotency_key ON jobs (idempotency_key) WHERE idempotency_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS admin_actions (
    id BIGSERIAL PRIMARY KEY,
    action TEXT NOT NULL,
    user_id BIGINT,
    admin_id BIGINT,
    payload JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE admin_actions IS 'Audit trail for manual admin actions (refunds, bans, grants)';

CREATE INDEX IF NOT EXISTS idx_admin_actions_user ON admin_actions (user_id);
CREATE INDEX IF NOT EXISTS idx_admin_actions_action ON admin_actions (action);

CREATE TABLE IF NOT EXISTS error_logs (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    user_id BIGINT,
    username TEXT,
    corr_id TEXT,
    job_id TEXT,
    model TEXT,
    size TEXT,
    status_code INTEGER,
    error_type TEXT,
    error_msg_short TEXT NOT NULL,
    refunded BOOLEAN DEFAULT FALSE,
    preflight_blocked BOOLEAN DEFAULT FALSE,
    preflight_reason TEXT,
    auto_sanitized BOOLEAN DEFAULT FALSE,
    sanitized_prompt TEXT,
    error_scope TEXT,
    error_json TEXT,
    job_status TEXT,
    reason TEXT,
    provider_error_code TEXT,
    provider_error_message TEXT,
    stage TEXT
);

COMMENT ON TABLE error_logs IS 'Structured provider errors captured for diagnostics';
CREATE INDEX IF NOT EXISTS idx_error_logs_ts ON error_logs (ts DESC);

CREATE TABLE IF NOT EXISTS archive_logs (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    corr_id TEXT NOT NULL,
    archive_status TEXT NOT NULL,
    channel_id BIGINT,
    message_id BIGINT,
    content_type TEXT,
    model_name TEXT,
    username TEXT,
    user_id BIGINT,
    caption_len INTEGER,
    file_size INTEGER,
    duration_seconds INTEGER,
    attempts INTEGER,
    error_short TEXT
);

COMMENT ON TABLE archive_logs IS 'Delivery attempts to archive channel';
CREATE INDEX IF NOT EXISTS idx_archive_logs_corr_id ON archive_logs (corr_id);
CREATE INDEX IF NOT EXISTS idx_archive_logs_status ON archive_logs (archive_status);

COMMIT;
