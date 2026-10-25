# Victorian Public Transport Patronage Analysis

Analysis of average daily public transport patronage in Victoria, Australia (2018 onwards), covering data cleaning, feature engineering, descriptive analytics and two simple machine learning models.

All work is in [`main.ipynb`](main.ipynb).

## Dataset

Input file: `avg_patronage.csv`

| Column | Description |
|---|---|
| `Year`, `Month`, `Month_name` | Reporting period |
| `Day_of_week` | Monday – Sunday |
| `Day_type` | e.g. Normal Weekday, School Holiday Weekday, Saturday, Sunday |
| `Mode` | Transport mode (Metro Train, Metro Bus, Metro Tram, Regional Bus, ...) |
| `Pax_daily` | Average daily passenger boardings |

## Notebook workflow

### 1. Data loading & cleaning
- Load the CSV with explicit missing-value markers.
- Cast `Year`/`Month` to integers and `Day_of_week`, `Day_type`, `Mode` to categories.
- Check for duplicates, nulls and anomalies in `Pax_daily`.
- Normalise category labels (strip whitespace, title case).

### 2. Feature engineering
- `MonthStart` – first date of each month (for time-series plotting).
- `Period` – `Pre-2020`, `2020-2021` (COVID period), `Post-2021`.
- `Is_Weekday` – 1 for Monday–Friday, 0 otherwise.
- `Rolling_3M` – 3-month rolling mean of `Pax_daily` per `Mode` and `Day_type`.

The cleaned dataset is saved to `patronage_cleaned.csv`.

### 3. Descriptive analytics (LO1)
- Mean/median daily patronage by `Mode`, `Day_type`, `Day_of_week` and `Period`.
- Monthly total passengers, overall and by mode.

### 4. Machine learning (LO2)
Categorical features are one-hot encoded and a **time-based split** is used: train on 2018–2023, test on 2024–2025.

1. **Linear Regression** – predicts `Pax_daily`.
   Evaluated with MAE, RMSE and R², plus an actual-vs-predicted plot and a coefficient chart.
2. **Decision Tree Classifier** (`max_depth=3`) – predicts whether a day is *Busy* (above that mode's median patronage in the training years) or *Quiet*.
   Evaluated with accuracy, a classification report, a confusion matrix and a plot of the tree.

## Requirements

- Python 3.10+
- pandas, numpy, matplotlib, scikit-learn, jupyter

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

## Running

Place `avg_patronage.csv` in the same folder as the notebook, then:

```bash
jupyter notebook main.ipynb
```

Run all cells from top to bottom.
