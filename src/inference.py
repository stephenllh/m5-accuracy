import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import lightgbm as lgb
from preprocess import process_calendar, process_prices


TR_LAST = 1913  # Last day used for training
MAX_LAGS = 57  # Maximum lag day


def create_ds():
    """Create dataset for predictions"""
    startDay = TR_LAST - MAX_LAGS

    numCols = [f"d_{day}" for day in range(startDay, TR_LAST + 1)]
    catCols = ["id", "item_id", "dept_id", "store_id", "cat_id", "state_id"]

    dtype = {numCol: "float32" for numCol in numCols}
    dtype.update({catCol: "category" for catCol in catCols if catCol != "id"})

    ds = pd.read_csv(
        "../input/sales_train_validation.csv",
        usecols=catCols + numCols,
        dtype=dtype,
    )

    for col in catCols:
        if col != "id":
            ds[col] = ds[col].cat.codes.astype("int16")
            ds[col] -= ds[col].min()

    for day in range(TR_LAST + 1, TR_LAST + 28 + 1):
        ds[f"d_{day}"] = np.nan

    ds = pd.melt(
        ds,
        id_vars=catCols,
        value_vars=[col for col in ds.columns if col.startswith("d_")],
        var_name="d",
        value_name="sales",
    )

    calendar = process_calendar()
    prices = process_prices()

    ds = ds.merge(calendar, on="d", copy=False)
    ds = ds.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], copy=False)

    return ds


def create_features(ds):
    dayLags = [7, 28]
    lagSalesCols = [f"lag_{dayLag}" for dayLag in dayLags]
    for dayLag, lagSalesCol in zip(dayLags, lagSalesCols):
        ds[lagSalesCol] = ds[["id", "sales"]].groupby("id")["sales"].shift(dayLag)

    windows = [7, 28]
    for window in windows:
        for dayLag, lagSalesCol in zip(dayLags, lagSalesCols):
            ds[f"rmean_{dayLag}_{window}"] = (
                ds[["id", lagSalesCol]]
                .groupby("id")[lagSalesCol]
                .transform(lambda x: x.rolling(window).mean())
            )

    dateFeatures = {
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
    }

    for featName, featFunc in dateFeatures.items():
        if featName in ds.columns:
            ds[featName] = ds[featName].astype("int16")
        else:
            ds[featName] = getattr(ds["date"].dt, featFunc).astype("int16")

    return ds


if __name__ == "__main__":
    trainCols = pd.read_csv("traincols.csv")
    m_lgb = lgb.load("model.lgb")

    fday = datetime(2016, 4, 25)
    sub = 0.0

    te = create_ds()
    cols = [f"F{i}" for i in range(1, 29)]

    for tdelta in range(0, 28):
        day = fday + timedelta(days=tdelta)
        print(tdelta, day)
        tst = te[
            (te["date"] >= day - timedelta(days=MAX_LAGS)) & (te["date"] <= day)
        ].copy()
        create_features(tst)
        tst = tst.loc[tst["date"] == day, trainCols]
        te.loc[te["date"] == day, "sales"] = m_lgb.predict(tst)

    sub = te.loc[te["date"] >= fday, ["id", "sales"]].copy()
    sub["F"] = [f"F{rank}" for rank in sub.groupby("id")["id"].cumcount() + 1]
    sub = sub.set_index(["id", "F"]).unstack()["sales"][cols].reset_index()
    sub.fillna(0.0, inplace=True)
    sub.sort_values("id", inplace=True)
    sub.reset_index(drop=True, inplace=True)
    sub.to_csv("submission_.csv", index=False)

    sub2 = sub.copy()
    sub2["id"] = sub2["id"].str.replace("validation$", "evaluation")
    sub = pd.concat([sub, sub2], axis=0, sort=False)
    sub.to_csv("submission.csv", index=False)
