-- Одноразовое исправление: установить флаги миграции для всех существующих пользователей
-- Это предотвратит повторное умножение кредитов при каждом деплое

BEGIN;

-- Установить economy_v2 = TRUE для всех пользователей, у которых он FALSE
-- Это означает, что они УЖЕ находятся в новой системе кредитов
UPDATE users
SET economy_v2 = TRUE,
    bonus_granted = TRUE,
    updated_at = NOW()
WHERE economy_v2 = FALSE OR bonus_granted = FALSE;

-- Проверить результат
SELECT
    COUNT(*) as total_users,
    COUNT(*) FILTER (WHERE economy_v2 = TRUE) as migrated_users,
    COUNT(*) FILTER (WHERE bonus_granted = TRUE) as bonus_granted_users
FROM users;

COMMIT;
