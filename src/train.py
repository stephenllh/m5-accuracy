import numpy as np
import lightgbm as lgb
from preprocess import process_ds


def train():
    np.random.seed(777)

    params = {
        "objective": "poisson",
        "metric": "rmse",
        "force_row_wise": True,
        "learning_rate": 0.075,
        "sub_row": 0.75,
        "bagging_freq": 1,
        "lambda_l2": 0.1,
        "verbosity": 1,
        "num_iterations": 1200,
        "num_leaves": 128,
        "min_data_in_leaf": 100,
    }

    # Define categorical features
    catFeats = [
        "item_id",
        "dept_id",
        "store_id",
        "cat_id",
        "state_id",
        "event_name_1",
        "event_name_2",
        "event_type_1",
        "event_type_2",
    ]

    X_train, y_train = process_ds()
    trainData = lgb.Dataset(
        X_train.loc[:-10000],
        label=y_train.loc[:-10000],
        categorical_feature=catFeats,
        free_raw_data=False,
    )
    validData = lgb.Dataset(
        X_train.loc[-10000:],
        label=y_train.loc[-10000:],
        categorical_feature=catFeats,
        free_raw_data=False,
    )

    m_lgb = lgb.train(params, trainData, valid_sets=[validData], verbose_eval=20)
    m_lgb.save_model("model.lgb")


if __name__ == "__main__":
    train()
