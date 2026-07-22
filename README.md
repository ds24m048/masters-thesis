# Day-Ahead Electricity Price Forecasting

Code for a master's thesis on forecasting the German day-ahead electricity price.
The project covers the full workflow: data acquisition, exploratory analysis, model
selection across statistical, machine-learning and deep-learning methods, and a
TreeSHAP-based explainability analysis of the final model.

## Repository structure

```
.
├── eda.ipynb              Exploratory data analysis of the target and features
├── modelling.ipynb        Feature selection, model training, validation and testing
├── explainability.ipynb   TreeSHAP analysis of the final XGBoost model
├── data/
│   ├── ENTSOE/
│   │   ├── entsoe.ipynb                    downloads price and load forecast
│   │   └── entsoe_price_and_loadfc.parquet
│   ├── Marktstammdatenregister/
│   │   ├── power_units.ipynb               parses the plant registry
│   │   ├── power_wind.parquet
│   │   └── power_solar.parquet
│   ├── NOAA GFS/
│   │   ├── noaa_gfs.py                      downloads and aggregates GFS forecasts
│   │   ├── aggregate_weather_fc.ipynb       merges the daily files
│   │   └── weather_forecast.parquet
│   └── *.csv                                saved hyperparameter-tuning results
└── models/                (gitignored) trained model files
```

## Data pipeline

Three independent sources are prepared separately and then joined on an hourly
`Europe/Berlin` index in `modelling.ipynb`. All processing is DST-aware. The target and
features cover 2023 to 2025 for the bidding zone DE-LU (Germany/Luxembourg).

### ENTSO-E: price and load (`data/ENTSOE/entsoe.ipynb`)

Downloads the day-ahead price (`da_price`) and the day-ahead load forecast (`load_fc`)
from the ENTSO-E Transparency Platform via the `entsoe-py` API (API key read from
`data/api_keys.txt`). The two series arrive at different and changing granularities:
the load forecast is quarter-hourly throughout, and the price switches from hourly to
quarter-hourly. Both are resampled to a common hourly grid (price by mean,
load by sum), and missing timestamps are filled by linear interpolation. Output:
`entsoe_price_and_loadfc.parquet`.

### Marktstammdatenregister: plant registry (`data/Marktstammdatenregister/power_units.ipynb`)

Parses the German plant registry (MaStR) XML bulk export with `iterparse`, keeping wind
and solar units together with their
coordinates and net rated capacity (`Nettonennleistung`). Solar records carry no
coordinates and some wind records are missing them, so the missing locations are filled
from the municipality key (`Gemeindeschluessel`) using centroids of the VG250 municipality
polygons pulled from the geodatenzentrum WFS service. Output: `power_wind.parquet` and
`power_solar.parquet`, one row per plant with location and capacity. These are not model
features; they exist only to weight the weather aggregation in the next step.

### NOAA GFS: weather forecasts (`data/NOAA GFS/noaa_gfs.py`, `aggregate_weather_fc.ipynb`)

`noaa_gfs.py` downloads GFS 0.25 degree GRIB2 forecasts from the NOAA public S3 bucket.
For each target day it selects, from the 06z model run, the 24 forecast hours that cover
one full local calendar day, accounting for the UTC offset (CET vs CEST). From each file
it extracts surface solar radiation (`ssrd`), total cloud cover (`tcc`), 2 m temperature
(`t2m`), surface pressure (`sp`), 2 m relative humidity (`rh2m`), and 100 m wind speed
(`U100`, computed as the magnitude of the u and v components).

The grid is then collapsed to a single national value per variable and hour by a
**capacity-weighted spatial average**. Using the plant registry from the previous step,
each plant is snapped to its nearest GFS grid point, the installed capacity per grid point
is summed and normalized to weights, and the weather field is averaged with those weights.
This is done twice: solar-capacity weights for the solar-relevant variables and
wind-capacity weights for the wind-relevant variables, so grid cells with more installed
capacity dominate the national average. Results are written as one Parquet file per day
(wind and solar separately).

`aggregate_weather_fc.ipynb` concatenates the daily files, converts to `Europe/Berlin`,
removes duplicate timestamps and fills three missing timestamps by linear interpolation. 
Output: `weather_forecast.parquet`.

## Method

Target transformation: `asinh(price / c)` with `c = median(|price|)` estimated on the
training data. The features are scaled with a `RobustScaler` fitted on the training data,
used for all models. All reported errors are back-transformed to EUR/MWh.

Feature set (11 features after selection): `load_fc`, `ssrd`, `U100`,
`price_lag_24h/48h/168h`, `hour` and `weekday` as sin/cos pairs, and `holiday_not_sunday`.
The lookback models additionally use a window of past prices: a  48-hour window
during model-type selection in the first trainings, and a window length tuned as a hyperparameter (over 12 to
168 hours) in the second trainings.

Chronological split, evaluated once on the test set:

| Split      | Period                | Use                                   |
|------------|-----------------------|---------------------------------------|
| Train      | 2023-01-01 to 2025-01 | model fitting and CV tuning           |
| Validation | 2025-01 to 2025-07    | first model-type selection            |
| Test       | 2025-07 to 2026-01    | final evaluation                      |

## Models

Baselines (naive persistence, ARIMA), SARIMAX, classical ML (XGBoost, Random Forest, SVR)
and deep learning (LSTM, BiLSTM).

The forecast is issued once per day for all 24 hours of the following day, so at forecast
time no actual price of the target day is known. This mirrors a realistic day-ahead
setting, since the day-ahead prices themselves are all determined on the preceding day. The ML and deep-learning models are
trained under different input strategies that differ in how they supply price history to
each target hour:

- **Point**: each target hour is predicted independently from its own features only
  (weather, load forecast, calendar, and the fixed price lags at 24/48/168 hours). No
  contiguous window of recent prices is used, and hours within a day do not depend on each
  other.
- **Fixed lookback**: in addition to the point features, a contiguous window of the most
  recent actually observed prices is added as input. The window is day-anchored: it ends at
  the last price known before the forecast is issued, so every hour of the target day sees
  the same real price history ending on the last our of the day before the delivery day.
- **Rollout**: the model predicts the target day hour by hour, and the lookback window
  advances into the horizon by feeding the model's own predictions back in as inputs for
  the later hours (recursive multi-step forecasting). For window positions that fall on the
  delivery day the model's own predictions are used, while positions on the previous day
  and earlier use the actual observed prices, which are already known at forecast time.

The classical ML models are run in all three strategies. The deep-learning models are
sequence models that require a price window as input, so they are run only in the fixed
lookback and rollout strategies, not point.

### Two-step model selection

The models are compared and finalized in two steps:

1. **Comparison.** All 15 candidate models are trained on the training set and compared on
   the validation set. In this step the price-history windows are fixed at 48 hours.
2. **Final training.** The best model overall (SARIMAX), together with the best-performing
   classical ML model (XGBoost) and the best-performing deep-learning sequence model
   (LSTM), are retrained on the combined train+val set. In this step the lookback window
   length for XGBoost and LSTM is tuned as a hyperparameter alongside the others, and each finalized model is
   evaluated once on the test set.


## Results

Validation results (step 1, all candidates):

![Validation results](assets/validation_results.png)


Test results (step 2, finalized models):

![Test results](assets/test_results.png)

Final model: **XGBoost with a fixed lookback** retrained on train+val with a 
24-hour window, with a test MAE of about 13.5 EUR/MWh wins against **SARIMAX** and **LSTM with a fixed lookback**.

## Explainability

`explainability.ipynb` loads the final model and applies exact TreeSHAP:
global importance, grouped and per-window-position importance, dependence plots,
temporal and seasonal contribution shares, and local
waterfall explanations for characteristic hours. Cyclic feature pairs (hour, weekday)
are collapsed into single features by summing their SHAP values.

## Running

The modelling notebook is written for Google Colab (each starts with a Drive-mount cell).
They can also run locally with the usual scientific Python stack
(pandas, numpy, scikit-learn, statsmodels, xgboost, tensorflow/keras, shap, matplotlib,
holidays). Run order: `eda.ipynb`, then `modelling.ipynb` (produces the model files),
then `explainability.ipynb`.

## Notes

- `data/api_keys.txt` and the `models/` folder (too big files) are gitignored. Model files are
  reproducible by re-running `modelling.ipynb`.
