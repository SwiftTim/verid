# 🚀 Velocity Bot: Start-up Guide (Daily Routine)

Follow these steps in order to ensure the AI, the Backend, and the Dashboard are perfectly synchronized.

## 1. Start the "Brain" (Google Colab)
* Open your **Deriv Predictor** notebook in Google Colab.
* Run the **Prediction Server** cell.
* Ensure you see the message: `🛰️ ANTIGRAVITY ENGINE ACTIVE: Waiting for ticks...`
* **Important:** Keep this tab open. If Colab disconnects, the bot will stop trading.

## 2. Start the "Heart" (Local Backend)
* Open your terminal in `/home/tim/Downloads/2026/der`.
* Run the following command:
  ```bash
  python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
  ```
* Check for these positive signals in the logs:
  - `✅ Connected to Deriv API`
  - `Deriv account connected | balance: $XXXX.XX`
  - `📊 Subscribed to 1HZ100V tick stream`

## 3. Open the "Eyes" (Dashboard)
* Open `/home/tim/Downloads/2026/der/dashboard.html` in your browser.
* **CRITICAL:** Press `Ctrl + F5` (Windows/Linux) or `Cmd + Shift + R` (Mac) to force the browser to load the latest fixes.
* Verify:
  - **Connection:** Should show `LIVE` (Green).
  - **Balance:** Should show your current Demo balance.
  - **Colab Memory:** Should start counting up toward 1000/1000.

## 4. Engaging the Bot
* Once **Colab Memory** reaches at least **30/1000**, you can start.
* Toggle **"Auto-Trade"** to ON (Green).
* Click **"START BOT"** in the Velocity Execution panel.
* **Observe:** The first trade should appear in the "Recent Trades" list within a few minutes.

---

## 🛠️ Troubleshooting (The "Quick Fixes")
* **If Balance says $... :** Refresh the dashboard (F5).
* **If Win Rate is 0% :** This is normal for the first few trades.
* **If Connection is RED :** Check your Internet and ensure your Deriv API Token hasn't expired.
* **If Memory stays at 0 :** Ensure the Colab cell is actually running and you can "ping" it from your local machine.

---
*Created by Antigravity AI - 2026-05-21*
