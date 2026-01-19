-- Проверка, существует ли пользователь в таблице users
-- Выполните этот SQL в Railway PostgreSQL

-- Проверить пользователя из логов (user_id=8167131744)
SELECT user_id, credits, created_at, updated_at
FROM users
WHERE user_id = 8167131744;

-- Если пустой результат - это проблема!
-- Тогда создайте пользователя вручную:
INSERT INTO users (user_id, credits, bonus_granted, created_at, updated_at)
VALUES (8167131744, 100, false, NOW(), NOW())
ON CONFLICT (user_id) DO NOTHING;

-- Проверить все записи jobs для этого пользователя
SELECT job_id, status, created_at, model, error
FROM jobs
WHERE user_id = 8167131744
ORDER BY created_at DESC
LIMIT 10;
