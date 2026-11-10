# roberta-sentiment-runner

Runs **financial RoBERTa-large sentiment** (`soleimanian/financial-roberta-large-sentiment`)
via GitHub Actions — part of a one-public-repo-per-tool pattern: each tool gets its own
repo, the commands to execute live in a repository variable, runs are triggered
manually, and outputs are collected as artifacts.

The model is a RoBERTa-large model fine-tuned for financial sentiment
(positive / negative / neutral). It is a larger counterpart to FinBERT
(`finbert-runner` in this org) and uses the same pilot sample so the two
models can be compared on identical texts.

## How to run an analysis

1. **Edit the commands**: repo **Settings → Secrets and variables → Actions →
   Variables** → set `COMMANDS` to the shell commands you want executed, e.g.
   ```
   mkdir -p results
   set -o pipefail; python scripts/roberta_sentiment.py --input samples/poul_headlines.txt 2>&1 | tee results/run.log
   ```
2. **Trigger manually**: **Actions** tab → *Run analysis* → *Run workflow*
   (or dispatch it through the GitHub API).
3. **Collect results**: download the `results` artifact from the finished run.
   It contains `sentiment.json` (per-item labels + probabilities, label counts,
   mean probabilities), `report.md`, and the run log.

## Design notes

- **Generic and multi-market.** `scripts/roberta_sentiment.py` scores arbitrary
  English financial texts — any stock, any market, any news source. Nothing in
  the workflow or script is tied to one exchange; the POUL headlines in
  `samples/` are only a pilot input. They are the same 13 texts used by
  `moos-labs/finbert-runner` (`samples/poul_headlines.txt`, copied unchanged).
- **Labels from the model itself.** Probabilities are mapped by the model's
  own `id2label`; the JSON output records both the canonical
  positive/negative/neutral probabilities and the raw label names/scores,
  so any difference in the model's label naming is visible, not hidden.
- **Cheap repeat runs.** The pip cache plus an `actions/cache` entry for
  `~/.cache/huggingface` (key `roberta-sentiment-model-v1`) mean later runs skip both the
  package downloads and the model download. The model is large (~1.6GB), so
  the cache matters more here than for smaller models.
- **CPU only.** `requirements.txt` pulls torch from the PyTorch CPU wheel index.
- **English only.** The model was fine-tuned on English financial text. Arabic
  headlines must be translated before scoring (the pilot headlines are English
  renderings of Arabic news items).
