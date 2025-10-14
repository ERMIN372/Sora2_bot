# Sora2_bot

Sora2_bot is a Telegram bot that connects to the Sora video generation API, manages
user jobs, and processes Telegram Stars payments.

## Prerequisites

- Python 3.10 or newer.
- A Telegram bot token issued by [@BotFather](https://t.me/BotFather).
- A Sora API key with access to the video generation endpoints.
- (Optional) Redis, if you plan to provide your own finite state machine (FSM)
  storage or rate limiting backend for Aiogram.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/Sora2_bot.git
   cd Sora2_bot
   ```
2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **(Optional) Install development tooling** such as `black`, `ruff`, or
   `pytest` if you plan to extend the bot and add automated tests.

## Environment configuration

Configuration is provided via environment variables. Copy `.env.example` to `.env`
and update the values for your environment. The application automatically loads
the `.env` file through [`python-dotenv`](https://github.com/theskumar/python-dotenv).

| Variable | Description |
| --- | --- |
| `BOT_TOKEN` | Telegram bot token obtained from BotFather. |
| `SORA_API_KEY` | API key for the Sora video generation service. |
| `SORA_API_URL` | Base URL for the Sora API (defaults to `https://api.sora.ai/v1`). |
| `DATABASE_PATH` | Path to the SQLite database file used for persistence. |
| `JOBS_CONCURRENCY` | Maximum number of concurrent generation jobs processed by the worker pool. |
| `MAX_JOBS_PER_USER` | Number of queued jobs allowed per user before new submissions are rejected. |
| `CREDITS_PER_PAYMENT` | Credits awarded to a user per successful payment. |
| `CREDITS_PER_GENERATION` | Credits consumed for each generation request. |
| `REQUEST_TIMEOUT` | Timeout (in seconds) for outbound API requests. |
| `REQUEST_RETRIES` | Number of retries for failed API requests. |
| `RETRY_BACKOFF` | Exponential backoff multiplier between retries. |
| `AIROGRAM_REDIS_URL` | Optional Redis connection URL for FSM storage or rate limiting. |

**Tips**

- Store secrets such as `BOT_TOKEN` and `SORA_API_KEY` in your secret manager in
  production environments and inject them as environment variables at runtime.
- When deploying to platforms that do not support `.env` files natively, export
  the variables or configure them through your platform's environment settings.
- Use separate databases (different `DATABASE_PATH` values) for development,
  staging, and production to avoid data collisions.

## Running the bot

Once the environment variables are configured you can start the bot with:

```bash
python main.py
```

The startup routine performs the following steps automatically:

1. Loads the configuration from the environment.
2. Connects to the SQLite database (creating it and running migrations if necessary).
3. Creates the Sora API client and the background job queue workers.
4. Starts polling the Telegram Bot API for updates.

Press `Ctrl+C` to stop the process. The shutdown hook drains the queue, closes
network sessions, and releases database connections.

To run the bot in the background in production, wrap the command in a process
supervisor such as `systemd`, `supervisord`, Docker, or a container orchestration
platform.

## Logging and observability

- Logging is configured globally with `logging.basicConfig(level=logging.INFO)`
  and emitted to standard output. Capture the process output in your container
  runtime or process supervisor for long-term storage.
- Job queue and Sora API interactions log informational events for successful
  transitions, warnings on retryable failures, and stack traces when unexpected
  exceptions are raised.
- To change the verbosity, adjust the `logging.basicConfig` call in `main.py`
  or wrap the bot in a small launcher script that configures logging before
  importing `main`.

## Operations runbooks

### Subscription and credit verification

1. Use the `/balance` command in Telegram to confirm the user's remaining credits.
2. Inspect the `payments` table (via SQLite or your admin tooling) to ensure the
   latest Telegram Stars payment was recorded with matching charge identifiers.
3. If the balance appears incorrect, re-run the `successful_payment_handler` by
   forwarding the original Telegram payment receipt to the bot while monitoring
   the logs for the `TelegramStarPaymentProcessor` activity.
4. Confirm that the user has not exceeded `MAX_JOBS_PER_USER`; queued jobs count
   against the limit until they complete or fail.

### Pricing adjustments

- Update `CREDITS_PER_PAYMENT` to control how many credits are awarded per
  successful Telegram Stars transaction.
- Update `CREDITS_PER_GENERATION` to control the cost of running a single video
  generation job.
- After changing either value, restart the bot so the new configuration is
  loaded. Existing database records remain unchanged, so document the effective
  date of pricing changes for support inquiries.

### Handling Sora outages and degraded performance

1. Monitor logs for repeated `Sora API error` messages or warning-level
   `Failed to poll job` entries.
2. During an outage, the job queue automatically retries failed polls with
   exponential backoff. Credits are refunded automatically for jobs that enter a
   `failed` or `errored` state.
3. If the outage is prolonged, temporarily pause new submissions by raising
   `CREDITS_PER_GENERATION` to a very high value or disabling the `/generate`
   command in your deployment branch.
4. Once the Sora service recovers, resume normal pricing and monitor the job
   queue until all pending jobs have transitioned to `completed`.

### Clearing queue backlogs

1. Check the `jobs` table for entries in the `queued` or `processing` state using
   your preferred SQLite client.
2. Increase `JOBS_CONCURRENCY` temporarily to allow more workers to process the
   backlog. Restart the bot to apply the change.
3. For jobs stuck in `processing` with no progress, manually re-enqueue them by
   updating their status to `queued` or delete them and refund credits
   (increment `credits` in the `users` table) as appropriate.
4. Confirm that the job queue reports `Recovered X pending jobs` without errors
   after a restart.

## Manual acceptance tests

Run these manual checks after significant changes to validate the core flows:

1. **Onboarding** – Send `/start` to the bot and verify the welcome message is
   returned and a new user record is created in the `users` table.
2. **Balance inquiry** – Send `/balance` and ensure the reported credit total
   matches the database value for the user.
3. **Prompt submission** – Send `/generate <prompt>` with sufficient credits and
   confirm you receive the queued confirmation and that a job record is created
   with status `queued`.
4. **Prompt validation** – Send `/generate` without a prompt and confirm the bot
   responds with the usage hint.
5. **Credit enforcement** – Reduce the user's credits below
   `CREDITS_PER_GENERATION` and verify `/generate` responds with the insufficient
   credits message.
6. **Concurrent job limit** – Create more than `MAX_JOBS_PER_USER` pending jobs
   and ensure the bot denies additional submissions.
7. **Payment processing** – Complete a Telegram Stars test payment and confirm
   the acknowledgement message appears and credits increase by
   `CREDITS_PER_PAYMENT`.
8. **Job completion notification** – Mock or wait for a job to complete and
   ensure the bot sends the completion message containing the download link.
9. **Job failure handling** – Simulate a failed job (by returning `failed` from
   the Sora API) and verify the failure message is sent and credits are refunded.
10. **Restart resilience** – Restart the bot while jobs are pending and confirm
    they are recovered and continue processing.
