-- Проверить сколько пользователей с неправильными флагами
-- Эти пользователи будут получать умножение кредитов при каждом деплое

SELECT
    COUNT(*) as total_users,
    COUNT(*) FILTER (WHERE economy_v2 = FALSE) as users_with_false_economy_v2,
    COUNT(*) FILTER (WHERE bonus_granted = FALSE) as users_with_false_bonus_granted,
    COUNT(*) FILTER (WHERE economy_v2 = FALSE OR bonus_granted = FALSE) as users_affected_by_migration
FROM users;

-- Показать первых 10 пользователей, которые будут затронуты миграцией
SELECT user_id, credits, economy_v2, bonus_granted, username
FROM users
WHERE economy_v2 = FALSE OR bonus_granted = FALSE
LIMIT 10;
