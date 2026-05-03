# Duolingo Gamification Sentiment

A feature-level examination of user sentiment toward gamification mechanics in
Duolingo, using ~20k filtered comments from Google Play, the Apple App Store,
Reddit (`r/duolingo`, `r/duolingomemes`), and Twitter/X. Final analytical
dataset: **N = 1,788** binary-sentiment comments; logistic regression with
**pseudo-R² = 0.3218**.

The full write-up is in [`docs/Final_Report.pdf`](docs/Final_Report.pdf).

## Headline findings

| Feature | β | p | Direction |
|---|---|---|---|
| Quests | +2.449 | <0.001 | Strongly positive |
| Chess mode | +1.265 | <0.001 | Strongly positive |
| Energy system | −2.063 | <0.001 | Strongly negative |
| XP system | −0.951 | <0.001 | Negative |

## Repository layout

```
.
├── analysis.ipynb              # consolidated end-to-end notebook
├── analysis.py                 # script form of the same pipeline
├── data/
│   ├── raw/                    # source scrapes (Apify)
│   │   ├── dataset_googleplay-reviews.csv
│   │   ├── dataset_appstore-appstore.csv
│   │   ├── dataset_tweet-twitter.csv
│   │   ├── dataset_reddit-scraper-r_duolingo.csv
│   │   └── dataset_reddit-scraper-pro_r_duolingomeme.csv
│   ├── processed/              # gamification filter + sampling + labeling templates
│   │   ├── final_gamification_dataset.csv
│   │   ├── sampled_3000_for_labeling.csv
│   │   ├── labeling_template_3000.csv
│   │   ├── labeling_template_3000_fixed.csv
│   │   └── labeling_template_3000_final.csv
│   └── analysis/               # labeled + feature-engineered analytical files
│       ├── labeling_final.csv  # manual gold standard + LLM labels (hand-edited)
│       ├── ready_to_run.csv
│       └── ready_to_run_num.csv
└── docs/
    ├── Final_Report.pdf
    ├── Final_Report.docx
    ├── Business_Model_Canvas.pdf
    ├── Business_Model_Canvas.docx
    └── Final_Presentation.pptx
```

## Pipeline

```
raw scrapes
  → text cleaning + 8-feature keyword flagging
  → filter to gamification-mentioning comments  (final_gamification_dataset.csv)
  → stratified sample of 3,000                  (sampled_3000_for_labeling.csv)
  → labeling templates (manual + LLM, 50-row batches)
  → labeling_final.csv  (manual gold + LLM, hand-edited)
  → feature engineering: platform / month / length controls
                                                (ready_to_run.csv → ready_to_run_num.csv)
  → logistic regression on positive-vs-negative sentiment
```

## Run

```bash
pip install pandas numpy statsmodels
python analysis.py
# or open analysis.ipynb
```

`labeling_final.csv` already contains the manual + LLM sentiment labels, so the
regression at the end of the notebook reproduces the report's headline numbers
without re-running the labeling step.

## Author

Muqiao Tao — `mt5201@nyu.edu` — NYU, Fall 2025.
