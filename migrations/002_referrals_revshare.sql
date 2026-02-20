-- Migration: Move referral tracking from Google Sheets to PostgreSQL
-- and switch from flat-bonus to RevShare (15% of every top-up).

BEGIN;

-- Permanent referral links: one referrer per referred user.
CREATE TABLE IF NOT EXISTS referrals (
    id            BIGSERIAL PRIMARY KEY,
    referrer_id   BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    referred_id   BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_referrals_referred UNIQUE (referred_id)
);

COMMENT ON TABLE referrals IS 'Permanent referrer-referred pairs (one referrer per user, lifetime bond)';
CREATE INDEX IF NOT EXISTS idx_referrals_referrer ON referrals (referrer_id);

-- Audit log of every RevShare payout.
CREATE TABLE IF NOT EXISTS referral_payouts (
    id                BIGSERIAL PRIMARY KEY,
    referral_id       BIGINT NOT NULL REFERENCES referrals(id) ON DELETE CASCADE,
    payment_ext_id    TEXT NOT NULL,
    payer_id          BIGINT NOT NULL,
    referrer_id       BIGINT NOT NULL,
    topup_amount_cp   INTEGER NOT NULL,
    payout_credits    INTEGER NOT NULL,
    payout_pct        NUMERIC(5, 2) NOT NULL DEFAULT 15.00,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE referral_payouts IS 'Audit trail of RevShare payouts (15% of each referral top-up)';
CREATE INDEX IF NOT EXISTS idx_referral_payouts_referrer ON referral_payouts (referrer_id);
CREATE INDEX IF NOT EXISTS idx_referral_payouts_payer ON referral_payouts (payer_id);
CREATE INDEX IF NOT EXISTS idx_referral_payouts_payment ON referral_payouts (payment_ext_id);

COMMIT;
