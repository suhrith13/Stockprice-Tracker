# Daily Stock Prediction Tracker

Every weekday, GitHub's servers run this automatically:

- **8:00 AM IST:** the model predicts each stock's closing price and logs it *before* the 9:15 open.
- **4:30 PM IST:** it fetches the real close, scores each prediction, and updates the charts.
- **The live webpage** always shows the latest results, re-checks the whole log in each visitor's browser, and compares the model against the naive guess "today = yesterday."

You don't need to keep your computer on. There's one log file and one set of charts, and both are updated in place every day.

It does **not** place trades. It only predicts, records and scores.

---

## One-time setup (about 10 minutes)

1. **Create a public repo** on GitHub, for example `stock-prediction-tracker`. Public repos get GitHub Actions and GitHub Pages for free.
2. **Upload these files** with the folder structure kept, including the hidden `.github/workflows/daily.yml`. The easiest way is from this folder on your computer:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/<your-username>/stock-prediction-tracker.git
   git push -u origin main
   ```
   *(GitHub's web uploader skips hidden folders, so use git or GitHub Desktop.)*
3. **Allow the workflow to save results:** go to Settings → Actions → General → Workflow permissions, choose **Read and write permissions**, and save.
4. **Turn on the webpage:** go to Settings → Pages. Under Source, pick **Deploy from a branch**, then branch `main` and folder **`/docs`**, and save. After a minute your page is live at `(https://suhrith13.github.io/Stockprice-Tracker/)`.
5. **Test it now:** open the Actions tab, select **Daily predictions**, then **Run workflow**, and pick `evaluate`. This confirms everything installs and runs. It's fine if there's nothing to evaluate yet. The first real predictions appear on the next weekday morning.

That's it. From then on it runs itself.

## Changing the stocks

Edit `config.json`. Use Yahoo Finance symbols: `.NS` for NSE (e.g. `RELIANCE.NS`) and `^NSEI` for the Nifty 50.

## Running it by hand (optional)

```bash
pip install -r requirements.txt
python tracker.py predict     # before 9:15 AM IST
python tracker.py evaluate    # after the close
python tracker.py report      # PNG charts in report/live/ (overwritten each time)
python tracker.py verify      # confirm no past prediction was edited
python tracker.py backtest --days 250   # simulated history, NOT live proof
```

Add `--demo` to any command to try it with simulated prices and no internet. To preview the webpage locally, run `python -m http.server` inside `docs/` and open `http://localhost:8000`.

## Why the results can be trusted

- **Timestamped before the outcome.** Each prediction records exactly when it was made. The tool **refuses** to log a prediction after the market has opened.
- **No overwriting.** A second prediction for the same day is skipped.
- **Hash chain.** Each row stores a SHA-256 fingerprint of itself plus the previous row, so editing any past prediction breaks the chain. The webpage re-checks every fingerprint in the visitor's own browser.
- **Independent witness.** Every daily update is a GitHub commit with its own timestamp, public for anyone to inspect.
- **Holidays** are detected automatically and excluded from scoring.

## Reading the results

- **Model avg error vs baseline:** the model must be *lower* than the baseline to show any skill.
- **Direction called correctly:** 50% is a coin flip.
- **Sample size:** the page says "too early to judge" until there are 60+ scored trading days. A good week proves nothing.

## Things to know

- GitHub's scheduled runs can start several minutes late at busy times. The 8:00 AM slot leaves over an hour of buffer before the open.
- Yahoo Finance occasionally rate-limits or fails. A failed day's run just shows as a red ✕ in the Actions tab, and the next evening's run catches up on anything still unscored.
- GitHub can pause scheduled workflows in repos with no activity for 60 days. The daily commits should keep it active, but if updates stop, check the Actions tab and re-enable the workflow.

*Educational and portfolio project. Not financial advice.*
