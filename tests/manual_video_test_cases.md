# Manual video delivery test cases

1. **Small video succeeds** – Gemini key configured, API available, generated file within Telegram video limit. Expect the bot to send the video via `sendVideo` and follow up with a summary message. No external links.
2. **Large video fallback to document** – Video exceeds the streaming limit but fits into Telegram document limit. Bot should upload via `sendDocument` and send the summary.
3. **SERVICE_DISABLED during download** – Simulate the API returning `SERVICE_DISABLED` while downloading. Bot retries once, then aborts with a configuration error message, logs the key mask, and refunds credits.
4. **Key mismatch detection** – Change the Gemini key between generation and download. Bot must stop the job, refund, log both masks, and notify the user.
5. **Telegram size constraint breach** – Generated file exceeds Telegram’s maximum document size. Bot should refuse delivery with a clear message and refund.
6. **Consecutive jobs same key** – Run three jobs sequentially for one user. Logs should consistently show the same Gemini key mask for submission, polling, download, and delivery.
