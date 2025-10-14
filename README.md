# Sora2_bot

Sora2_bot is a Telegram bot that connects to the Sora video generation API, manages
user jobs, and processes Telegram Stars payments.

## Installation

1. Create a Python virtual environment.
2. Activate the environment.
3. Install the dependencies: `pip install -r requirements.txt`.

## Configuration

Configuration is provided via environment variables. Copy `.env.example` to `.env`
and update the values for your environment. The application automatically loads
the `.env` file through [`python-dotenv`](https://github.com/theskumar/python-dotenv).

| Variable | Description |
| --- | --- |
| `BOT_TOKEN` | Telegram bot token obtained from BotFather. |
| `SORA_API_KEY` | API key for the Sora video generation service. |
| `SORA_API_URL` | Base URL for the Sora API. |
| `DATABASE_PATH` | Path to the SQLite database file used for persistence. |
| `JOBS_CONCURRENCY` | Maximum number of concurrent generation jobs. |
| `MAX_JOBS_PER_USER` | Number of queued jobs allowed per user. |
| `CREDITS_PER_PAYMENT` | Credits awarded to a user per successful payment. |
| `CREDITS_PER_GENERATION` | Credits consumed for each generation request. |
| `REQUEST_TIMEOUT` | Timeout (in seconds) for outbound API requests. |
| `REQUEST_RETRIES` | Number of retries for failed API requests. |
| `RETRY_BACKOFF` | Exponential backoff multiplier between retries. |
| `AIROGRAM_REDIS_URL` | Optional Redis connection URL for FSM storage or rate limiting. |

## Running the bot

Once configured, start the bot with:

```bash
python main.py
```
