import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import precision_score
sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")
sp500 = sp500.drop(columns=["Dividends", "Stock Splits"])
sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
sp500 = sp500.loc["1990-01-01":].copy()
predictors = ["Close", "Volume", "Open", "High", "Low"]
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined
def backtest(data, model, predictors, start=2500, step=250):
    all = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        preds = predict(train, test, predictors, model)
        all.append(preds)
    return pd.concat(all)
predictions = backtest(sp500, model, predictors)
horizons = [2, 5, 60, 250, 1000]
new_predictors = []
for h in horizons:
    rollingAvg = sp500.rolling(h).mean()
    ratio_column = f"Close_ratio_{h}"
    sp500[ratio_column] = sp500["Close"] / rollingAvg["Close"]

    trend_column = f"Trend_{h}"
    sp500[trend_column] = sp500.shift(1).rolling(h).sum()["Target"]

    new_predictors += [ratio_column, trend_column]
sp500 = sp500.dropna(subset=sp500.columns[sp500.columns != "Tomorrow"])
predictions = backtest(sp500, model, new_predictors)
print("Precision score:", precision_score(predictions["Target"], predictions["Predictions"]))
latest_data = sp500.iloc[-1:][predictors]
model.fit(sp500[predictors], sp500["Target"])
tomorrow_prediction = model.predict(latest_data)
if tomorrow_prediction == 1:
    print("The S&P 500 is predicted to rise tomorrow.")
else:
    print("The S&P 500 is predicted to not rise tomorrow.")