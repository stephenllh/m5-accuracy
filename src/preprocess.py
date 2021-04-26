import pandas as pd


def process_calendar():
    calendarDTypes = {
        "event_name_1": "category",
        "event_name_2": "category",
        "event_type_1": "category",
        "event_type_2": "category",
        "weekday": "category",
        "wm_yr_wk": "int16",
        "wday": "int16",
        "month": "int16",
        "year": "int16",
        "snap_CA": "float32",
        "snap_TX": "float32",
        "snap_WI": "float32",
    }

    # Read csv file
    calendar = pd.read_csv("../input/calendar.csv", dtype=calendarDTypes)

    calendar["date"] = pd.to_datetime(calendar["date"])

    # Transform categorical features into integers
    for col, colDType in calendarDTypes.items():
        if colDType == "category":
            calendar[col] = calendar[col].cat.codes.astype("int16")
            calendar[col] -= calendar[col].min()

    return calendar


def process_prices():
    # Correct data types for "sell_prices.csv"
    priceDTypes = {
        "store_id": "category",
        "item_id": "category",
        "wm_yr_wk": "int16",
        "sell_price": "float32",
    }

    # Read csv file
    prices = pd.read_csv("../input/sell_prices.csv", dtype=priceDTypes)

    # Transform categorical features into integers
    for col, colDType in priceDTypes.items():
        if colDType == "category":
            prices[col] = prices[col].cat.codes.astype("int16")
            prices[col] -= prices[col].min()

    return prices


def process_ds():
    firstDay = 250
    lastDay = 1913

    # Use x sales days (columns) for training
    numCols = [f"d_{day}" for day in range(firstDay, lastDay + 1)]

    # Define all categorical columns
    catCols = ["id", "item_id", "dept_id", "store_id", "cat_id", "state_id"]

    # Define the correct data types for "sales_train_validation.csv"
    dtype = {numCol: "float32" for numCol in numCols}
    dtype.update({catCol: "category" for catCol in catCols if catCol != "id"})

    # Read csv file
    ds = pd.read_csv(
        "../input/sales_train_validation.csv",
        usecols=catCols + numCols,
        dtype=dtype,
    )

    # Transform categorical features into integers
    for col in catCols:
        if col != "id":
            ds[col] = ds[col].cat.codes.astype("int16")
            ds[col] -= ds[col].min()

    ds = pd.melt(
        ds,
        id_vars=catCols,
        value_vars=[col for col in ds.columns if col.startswith("d_")],
        var_name="d",
        value_name="sales",
    )

    calendar = process_calendar()
    prices = process_prices()
    # Merge "ds" with "calendar" and "prices" dataframe
    ds = ds.merge(calendar, on="d", copy=False)
    ds = ds.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], copy=False)

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

    # Remove all rows with NaN value
    ds.dropna(inplace=True)

    # Define columns that need to be removed
    unusedCols = ["id", "date", "sales", "d", "wm_yr_wk", "weekday"]
    trainCols = ds.columns[~ds.columns.isin(unusedCols)]
    trainCols.to_csv("traincols.csv", index=False)
    X_train = ds[trainCols]
    y_train = ds["sales"]

    return X_train, y_train
