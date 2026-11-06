# Freight Rate Prediction 

Predicts shipment `posted_rate` using the provided January–October 2025 labeled data. The model is evaluated on future months and applied to the supplied November–December rows.

## Files
- `freight/features.py`: dataset-specific cleaning and feature engineering
- `freight/model.py`: robust log-linear baseline, LightGBM corrections, lane adjustments
- `run_cv.py`: all three rolling chronological validation folds, **recomputed** every run
- `train_predict.py`: trains on all 48,000 labeled rows and writes both CSV prediction files
- `score.py`: unchanged assessment-provided output validator and December chart generator
- `data/`: supplied CSV inputs
- `outputs/Freight_Rate_Report.pdf`: summary of the methodology and verified validation results

## Run from the project folder

```bash
python -m pip install -r requirements.txt
python run_cv.py
python train_predict.py
python score.py --predictions validation_predictions.csv --december-predictions outputs/december_chart_predictions.csv --output-dir outputs/scorer_results
```

These scripts generate `outputs/cv_results.csv`, `validation_predictions.csv`, `outputs/december_chart_predictions.csv`, and `outputs/scorer_results/candidate_december.png`. 

## Approach
The first stage fits robust linear regression to log freight rates, with distance, weight, equipment and date-level inputs. The second stage fits LightGBM to the residual log-price errors. The last stage adds a small, shrunken correction for pickup/delivery lanes. Two trend specifications are averaged because extrapolating November–December is uncertain. Values with very large log residuals are excluded from fitting but not validation.

Three chronological validation windows are used. 

## Important modeling assumptions
- Negative weights appear to be sign mistakes; missing weights use equipment medians **from the training fold**.
- Daily market-index averages are computed from the supplied features for the relevant period, not from prediction targets. This assumes unlabeled same-day market inputs are available.
- Geographic coordinates are city-level descriptors, not precise pickup/delivery addresses.
- Some large posted rates may be genuine; robust trimming is a modeling choice, not proof of incorrect labels.
