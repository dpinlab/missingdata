"""
Preprocessing func for the dataset PhysionNet2012.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pandas as pd
import tsdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from ..utils.logging import logger, print_final_dataset_info
from ..utils.missingness import create_missingness


def preprocess_physionet2012(
    subset,
    rate,
    normalization,
    pattern: str = "point",
    features: list = None,
    **kwargs,
) -> dict:
    """Load and preprocess the dataset PhysionNet2012.

    Parameters
    ----------
    subset:
        The name of the subset dataset to be loaded.
        Must be one of ['all', 'set-a', 'set-b', 'set-c'].

    rate:
        The missing rate.

    pattern:
        The missing pattern to apply to the dataset.
        Must be one of ['point', 'subseq', 'block'].

    features:
        The features to be used in the dataset.
        If None, all features except the static features will be used.

    Returns
    -------
    processed_dataset :
        A dictionary containing the processed PhysionNet2012.

    """

    def apply_func(df_temp):  # pad and truncate to set the max length of samples as 48
        missing = list(set(range(0, 48)).difference(set(df_temp["Time"])))
        missing_part = pd.DataFrame({"Time": missing})
        df_temp = pd.concat(
            [df_temp, missing_part], ignore_index=False, sort=False
        )  # pad the sample's length to 48 if it doesn't have enough time steps
        df_temp = df_temp.set_index("Time").sort_index().reset_index()
        df_temp = df_temp.iloc[:48]  # truncate
        return df_temp

    all_subsets = ["all", "set-a", "set-b", "set-c"]
    assert (
        subset.lower() in all_subsets
    ), f"subset should be one of {all_subsets}, but got {subset}"
    assert 0 <= rate < 1, f"rate must be in [0, 1), but got {rate}"

    # read the raw data
    data = tsdb.load("physionet_2012")
    all_features = set(data["set-a"].columns)
    data["static_features"].remove("ICUType")  # keep ICUType for now

    if subset != "all":
        df = data[subset]
        X = df.reset_index(drop=True)
        unique_ids = df["RecordID"].unique()
        y = data[f"outcomes-{subset.split('-')[-1]}"]
        y = y.loc[unique_ids]
    else:
        df = pd.concat([data["set-a"], data["set-b"], data["set-c"]], sort=True)
        X = df.reset_index(drop=True)
        unique_ids = df["RecordID"].unique()
        y = pd.concat([data["outcomes-a"], data["outcomes-b"], data["outcomes-c"]])
        y = y.loc[unique_ids]

    def drop_static_features(X):
        if (
            features is None
        ):  # if features are not specified, we use all features except the static features, e.g. age
            X = X.drop(data["static_features"], axis=1)
        else:  # if features are specified by users, only use the specified features
            # check if the given features are valid
            features_set = set(features)
            if not all_features.issuperset(features_set):
                intersection_feats = all_features.intersection(features_set)
                difference = features_set.difference(intersection_feats)
                raise ValueError(
                    f"Given features contain invalid features that not in the dataset: {difference}"
                )
            # check if the given features contain necessary features for preprocessing
            if "RecordID" not in features:
                features.append("RecordID")
            if "ICUType" not in features:
                features.append("ICUType")
            if "Time" not in features:
                features.append("Time")
            # select the specified features finally
            X = X[features]

        X = X.drop(["level_1", "ICUType"], axis=1)

        return X
    
    X = X.groupby("RecordID").apply(apply_func)
    X = X.drop("RecordID", axis=1)
    X = X.reset_index()
    ICUType = X[["RecordID", "ICUType"]].set_index("RecordID").dropna()

    # PhysioNet2012 is an imbalanced dataset, hence, we separate positive and negative samples here for later splitting
    # This is to ensure positive and negative ratios are similar in train/val/test sets
    all_recordID = X["RecordID"].unique()
    positive = (y == 1)["In-hospital_death"]
    positive_sample_IDs = y.loc[positive].index.to_list()
    negative_sample_IDs = np.setxor1d(all_recordID, positive_sample_IDs)
    assert len(positive_sample_IDs) + len(negative_sample_IDs) == len(all_recordID)

    # split the dataset into the train, val, and test sets
    train_positive_set_ids, test_positive_set_ids = train_test_split(
        positive_sample_IDs, test_size=0.2
    )
    train_positive_set_ids, val_positive_set_ids = train_test_split(
        train_positive_set_ids, test_size=0.2
    )
    train_negative_set_ids, test_negative_set_ids = train_test_split(
        negative_sample_IDs, test_size=0.2
    )
    train_negative_set_ids, val_negative_set_ids = train_test_split(
        train_negative_set_ids, test_size=0.2
    )
    train_set_ids = np.concatenate([train_positive_set_ids, train_negative_set_ids])
    val_set_ids = np.concatenate([val_positive_set_ids, val_negative_set_ids])
    test_set_ids = np.concatenate([test_positive_set_ids, test_negative_set_ids])
    train_set_ids.sort()
    val_set_ids.sort()
    test_set_ids.sort()
    train_set = X[X["RecordID"].isin(train_set_ids)].sort_values(["RecordID", "Time"])
    val_set = X[X["RecordID"].isin(val_set_ids)].sort_values(["RecordID", "Time"])
    test_set = X[X["RecordID"].isin(test_set_ids)].sort_values(["RecordID", "Time"])

    #dividir o conjunto de teste nos subgrupos

    #Divisão por gênero

    female_gender_test_ids = test_set[test_set["Gender"] == 0.0]
    female_gender_test_ids  = female_gender_test_ids["RecordID"]
    female_gender_test = test_set[test_set["RecordID"].isin(female_gender_test_ids)]


    male_gender_test_ids = test_set[test_set["Gender"] == 1.0]
    male_gender_test_ids  = male_gender_test_ids["RecordID"]
    male_gender_test = test_set[test_set["RecordID"].isin(male_gender_test_ids)]

    undefined_gender_test_ids = test_set[test_set["Gender"] == -1.0]
    undefined_gender_test_ids  = undefined_gender_test_ids["RecordID"]
    undefined_gender_test = test_set[test_set["RecordID"].isin(undefined_gender_test_ids)]

    #Divisão por idade

    more_than_or_equal_to_65_test_ids = test_set[test_set["Age"] >= 65]
    more_than_or_equal_to_65_test_ids = more_than_or_equal_to_65_test_ids[more_than_or_equal_to_65_test_ids["Time"] == 0.0]
    more_than_or_equal_to_65_test_ids = more_than_or_equal_to_65_test_ids["RecordID"]
    more_than_or_equal_to_65_test = test_set[test_set["RecordID"].isin(more_than_or_equal_to_65_test_ids)]

    less_than_65_test_ids = test_set[test_set["Age"] < 65]
    less_than_65_test_ids = less_than_65_test_ids[less_than_65_test_ids["Time"] == 0.0]
    less_than_65_test_ids = less_than_65_test_ids["RecordID"]
    less_than_65_test = test_set[test_set["RecordID"].isin(less_than_65_test_ids)]

    #Divisão por ICUType

    #ICUType_1 = test_set[test_set['ICUType'] == 1.0]
    #ICUType_1 = ICUType_1[ICUType_1["Time"] == 0.0]
    #ICUType_1_ids = ICUType_1["RecordID"]
    #ICUType_1_test = test_set[test_set["RecordID"].isin(ICUType_1_ids)]

    #ICUType_2 = test_set[test_set['ICUType'] == 2.0]
    #ICUType_2 = ICUType_2[ICUType_2["Time"] == 0.0]
    #ICUType2_ids = ICUType_2["RecordID"]
    #ICUType_2_test = test_set[test_set["RecordID"].isin(ICUType2_ids)]

    #ICUType_3 = test_set[test_set['ICUType'] == 3.0]
    #ICUType_3 = ICUType_3[ICUType_3["Time"] == 0.0]
    #ICUType_3_ids = ICUType_3["RecordID"]
    #ICUType_3_test = test_set[test_set["RecordID"].isin(ICUType_3_ids)]

    #ICUType_4 = test_set[test_set['ICUType'] == 4.0]
    #ICUType_4 = ICUType_4[ICUType_4["Time"] == 0.0]
    #ICUType_4_ids = ICUType_4["RecordID"]
    #ICUType_4_test = test_set[test_set["RecordID"].isin(ICUType_4_ids)]

    #Divisão por IMC

    def classify_BMI(BMI):
        if BMI <= 18.5:
            return "Baixo peso"
        elif BMI >= 18.6 and BMI <= 24.9:
            return "Peso normal"
        elif BMI >= 25 and BMI <= 29.9:
            return "Sobrepeso"
        elif BMI >= 30:
            return "Obesity"
   
        
    filtered_test = test_set[(test_set['Height'] != -1) & (test_set['Weight'] != -1) & (test_set['Height'].notna()) & (test_set['Weight'].notna())] 

    filtered_test_metros = filtered_test.copy()
    filtered_test_metros["Height"] = filtered_test_metros["Height"]/100

    bmi_data_test = filtered_test_metros
    bmi_data_test["BMI"] = round(filtered_test_metros["Weight"] / (filtered_test_metros["Height"]**2), 1)
    bmi_data_test["Classificacao"] = bmi_data_test["BMI"].apply(classify_BMI)

    bmi_data_test = bmi_data_test.groupby("RecordID").first().reset_index()

    classificacao_undefined_ids = bmi_data_test["RecordID"]
    classificacao_undefined_test = test_set[~test_set["RecordID"].isin(classificacao_undefined_ids)]
    classificacao_undefined_ids = classificacao_undefined_test.copy()
    classificacao_undefined_ids = classificacao_undefined_ids.groupby("RecordID").first().reset_index()
    classificacao_undefined_ids = classificacao_undefined_ids["RecordID"]

    classificacao_baixo_peso_ids = bmi_data_test[bmi_data_test["Classificacao"] == "Baixo peso"]
    classificacao_baixo_peso_ids = classificacao_baixo_peso_ids["RecordID"]
    classificacao_baixo_peso_test = test_set[test_set["RecordID"].isin(classificacao_baixo_peso_ids)]

    classificacao_normal_peso_ids = bmi_data_test[bmi_data_test["Classificacao"] == "Peso normal"]
    classificacao_normal_peso_ids = classificacao_normal_peso_ids["RecordID"]
    classificacao_normal_peso_test = test_set[test_set["RecordID"].isin(classificacao_normal_peso_ids)]

    classificacao_sobrepeso_ids = bmi_data_test[bmi_data_test["Classificacao"] == "Sobrepeso"]
    classificacao_sobrepeso_ids = classificacao_sobrepeso_ids["RecordID"]
    classificacao_sobrepeso_test = test_set[test_set["RecordID"].isin(classificacao_sobrepeso_ids)]

    classificacao_obesidade_ids = bmi_data_test[bmi_data_test["Classificacao"] == "Obesity"]
    classificacao_obesidade_ids = classificacao_obesidade_ids["RecordID"]
    classificacao_obesidade_test = test_set[test_set["RecordID"].isin(classificacao_obesidade_ids)]

    #classificacao_obesidade_1_ids = bmi_data_test[bmi_data_test["Classificacao"] == "Obesidade grau 1"]
   #classificacao_obesidade_1_ids = classificacao_obesidade_1_ids["RecordID"]
    #classificacao_obesidade_1_test = test_set[test_set["RecordID"].isin(classificacao_obesidade_1_ids)]

    #classificacao_obesidade_2_ids = bmi_data_test[bmi_data_test["Classificacao"] == "Obesidade grau 2"]
    #classificacao_obesidade_2_ids = classificacao_obesidade_2_ids["RecordID"]
    #classificacao_obesidade_2_test = test_set[test_set["RecordID"].isin(classificacao_obesidade_2_ids)]

    #classificacao_obesidade_3_ids = bmi_data_test[bmi_data_test["Classificacao"] == "Obesidade grau 3"]
    #classificacao_obesidade_3_ids = classificacao_obesidade_3_ids["RecordID"]
    #classificacao_obesidade_3_test = test_set[test_set["RecordID"].isin(classificacao_obesidade_3_ids)]

    train_set = drop_static_features(train_set)
    val_set = drop_static_features(val_set)
    test_set = drop_static_features(test_set)
    female_gender_test = drop_static_features(female_gender_test)
    male_gender_test = drop_static_features(male_gender_test)
    undefined_gender_test = drop_static_features(undefined_gender_test)
    more_than_or_equal_to_65_test = drop_static_features(more_than_or_equal_to_65_test)
    less_than_65_test = drop_static_features(less_than_65_test)
    #ICUType_1_test = drop_static_features(ICUType_1_test)
    #ICUType_2_test = drop_static_features(ICUType_2_test)
    #ICUType_3_test = drop_static_features(ICUType_3_test)
    #ICUType_4_test = drop_static_features(ICUType_4_test)
    classificacao_undefined_test = drop_static_features(classificacao_undefined_test)
    classificacao_baixo_peso_test = drop_static_features(classificacao_baixo_peso_test)
    classificacao_normal_peso_test = drop_static_features(classificacao_normal_peso_test)
    classificacao_sobrepeso_test = drop_static_features(classificacao_sobrepeso_test)
    classificacao_obesidade_test = drop_static_features(classificacao_obesidade_test)
    #classificacao_obesidade_1_test = drop_static_features(classificacao_obesidade_1_test)
    #classificacao_obesidade_2_test = drop_static_features(classificacao_obesidade_2_test)
    #classificacao_obesidade_3_test = drop_static_features(classificacao_obesidade_3_test)
    
    # if (
    #      features is None
    # ):  # if features are not specified, we use all features except the static features, e.g. age
    #      X = X.drop(data["static_features"], axis=1)
    # else:  # if features are specified by users, only use the specified features
    #      # check if the given features are valid
    #      features_set = set(features)
    #      if not all_features.issuperset(features_set):
    #          intersection_feats = all_features.intersection(features_set)
    #          difference = features_set.difference(intersection_feats)
    #          raise ValueError(
    #              f"Given features contain invalid features that not in the dataset: {difference}"
    #          )
    #      # check if the given features contain necessary features for preprocessing
    #      if "RecordID" not in features:
    #          features.append("RecordID")
    #      if "ICUType" not in features:
    #          features.append("ICUType")
    #      if "Time" not in features:
    #          features.append("Time")
    #      # select the specified features finally
    #      X = X[features]

    # X = X.drop(["level_1", "ICUType"], axis=1)

    # remove useless columns and turn into numpy arrays
    train_set = train_set.drop(["RecordID", "Time"], axis=1)
    val_set = val_set.drop(["RecordID", "Time"], axis=1)
    test_set = test_set.drop(["RecordID", "Time"], axis=1)
    female_gender_test = female_gender_test.drop(["RecordID", "Time"], axis=1)
    male_gender_test = male_gender_test.drop(["RecordID", "Time"], axis=1)
    undefined_gender_test = undefined_gender_test.drop(["RecordID", "Time"], axis=1)
    more_than_or_equal_to_65_test = more_than_or_equal_to_65_test.drop(["RecordID", "Time"], axis=1)
    less_than_65_test = less_than_65_test.drop(["RecordID", "Time"], axis=1)
    #ICUType_1_test = ICUType_1_test.drop(["RecordID", "Time"], axis=1)
    #ICUType_2_test = ICUType_2_test.drop(["RecordID", "Time"], axis=1)
    #ICUType_3_test = ICUType_3_test.drop(["RecordID", "Time"], axis=1)
    #ICUType_4_test = ICUType_4_test.drop(["RecordID", "Time"], axis=1)
    classificacao_undefined_test = classificacao_undefined_test.drop(["RecordID", "Time"], axis=1)
    classificacao_baixo_peso_test = classificacao_baixo_peso_test.drop(["RecordID", "Time"], axis=1)
    classificacao_normal_peso_test = classificacao_normal_peso_test.drop(["RecordID", "Time"], axis=1)
    classificacao_sobrepeso_test = classificacao_sobrepeso_test.drop(["RecordID", "Time"], axis=1)
    classificacao_obesidade_test = classificacao_obesidade_test.drop(["RecordID", "Time"], axis=1)
    #classificacao_obesidade_1_test = classificacao_obesidade_1_test.drop(["RecordID", "Time"], axis=1)
    #classificacao_obesidade_2_test = classificacao_obesidade_2_test.drop(["RecordID", "Time"], axis=1)
    #classificacao_obesidade_3_test = classificacao_obesidade_3_test.drop(["RecordID", "Time"], axis=1)

    train_X, val_X, test_X, female_gender_test_X, male_gender_test_X, undefined_gender_test_X, more_than_or_equal_to_65_test_X, less_than_65_test_X, classificacao_undefined_test_X, classificacao_baixo_peso_test_X, classificacao_normal_peso_test_X, classificacao_sobrepeso_test_X, classificacao_obesidade_test_X= (
        train_set.to_numpy(),
        val_set.to_numpy(),
        test_set.to_numpy(),
        female_gender_test.to_numpy(),
        male_gender_test.to_numpy(),
        undefined_gender_test.to_numpy(),
        more_than_or_equal_to_65_test.to_numpy(),
        less_than_65_test.to_numpy(),
        #ICUType_1_test.to_numpy(),
        #ICUType_2_test.to_numpy(),
        #ICUType_3_test.to_numpy(),
        #ICUType_4_test.to_numpy(),
        classificacao_undefined_test.to_numpy(),
        classificacao_baixo_peso_test.to_numpy(),
        classificacao_normal_peso_test.to_numpy(),
        classificacao_sobrepeso_test.to_numpy(),
        classificacao_obesidade_test.to_numpy(),
        #classificacao_obesidade_2_test.to_numpy(),
        #classificacao_obesidade_3_test.to_numpy()
    )

    # normalization - StandardScale/MinMaxScaler

    if(normalization == 1):
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        val_X = scaler.transform(val_X)
        test_X = scaler.transform(test_X)
        female_gender_test_X = scaler.transform(female_gender_test_X)
        male_gender_test_X = scaler.transform(male_gender_test_X)
        undefined_gender_test_X = scaler.transform(undefined_gender_test_X)
        more_than_or_equal_to_65_test_X = scaler.transform(more_than_or_equal_to_65_test_X) 
        less_than_65_test_X = scaler.transform(less_than_65_test_X)
        #ICUType_1_test_X = scaler.transform(ICUType_1_test_X)
        #ICUType_2_test_X = scaler.transform(ICUType_2_test_X)
        #ICUType_3_test_X = scaler.transform(ICUType_3_test_X)
        #ICUType_4_test_X = scaler.transform(ICUType_4_test_X)
        classificacao_undefined_test_X = scaler.transform(classificacao_undefined_test_X)
        classificacao_baixo_peso_test_X = scaler.transform(classificacao_baixo_peso_test_X)
        classificacao_normal_peso_test_X = scaler.transform(classificacao_normal_peso_test_X)
        classificacao_sobrepeso_test_X = scaler.transform(classificacao_sobrepeso_test_X)
        classificacao_obesidade_test_X = scaler.transform(classificacao_obesidade_test_X)
        #classificacao_obesidade_2_test_X = scaler.transform(classificacao_obesidade_2_test_X)
        #classificacao_obesidade_3_test_X = scaler.transform(classificacao_obesidade_3_test_X)
        
    
    elif(normalization == 2):
        scaler = MinMaxScaler(feature_range=(0,1), clip=True)
        train_X = scaler.fit_transform(train_X)
        val_X = scaler.transform(val_X)
        test_X = scaler.transform(test_X)
        female_gender_test_X = scaler.transform(female_gender_test_X)
        male_gender_test_X = scaler.transform(male_gender_test_X)
        undefined_gender_test_X = scaler.transform(undefined_gender_test_X)
        more_than_or_equal_to_65_test_X = scaler.transform(more_than_or_equal_to_65_test_X) 
        less_than_65_test_X = scaler.transform(less_than_65_test_X)
        #ICUType_1_test_X = scaler.transform(ICUType_1_test_X)
        #ICUType_2_test_X = scaler.transform(ICUType_2_test_X)
        #ICUType_3_test_X = scaler.transform(ICUType_3_test_X)
        #ICUType_4_test_X = scaler.transform(ICUType_4_test_X)
        classificacao_undefined_test_X = scaler.transform(classificacao_undefined_test_X)
        classificacao_baixo_peso_test_X = scaler.transform(classificacao_baixo_peso_test_X)
        classificacao_normal_peso_test_X = scaler.transform(classificacao_normal_peso_test_X)
        classificacao_sobrepeso_test_X = scaler.transform(classificacao_sobrepeso_test_X)
        classificacao_obesidade_test_X = scaler.transform(classificacao_obesidade_test_X)
        #classificacao_obesidade_2_test_X = scaler.transform(classificacao_obesidade_2_test_X)
        #classificacao_obesidade_3_test_X = scaler.transform(classificacao_obesidade_3_test_X)

    # reshape into time series samples
    train_X = train_X.reshape(len(train_set_ids), 48, -1)
    val_X = val_X.reshape(len(val_set_ids), 48, -1)
    test_X = test_X.reshape(len(test_set_ids), 48, -1)
    female_gender_test_X = female_gender_test_X.reshape(len(female_gender_test_ids), 48, -1)
    male_gender_test_X = male_gender_test_X.reshape(len(male_gender_test_ids), 48, -1)
    undefined_gender_test_X = undefined_gender_test_X.reshape(len(undefined_gender_test_ids), 48, -1)
    more_than_or_equal_to_65_test_X = more_than_or_equal_to_65_test_X.reshape(len(more_than_or_equal_to_65_test_ids), 48, -1)
    less_than_65_test_X = less_than_65_test_X.reshape(len(less_than_65_test_ids), 48, -1)
    #ICUType_1_test_X = ICUType_1_test_X.reshape(len(ICUType_1_ids), 48, -1)
    #ICUType_2_test_X = ICUType_2_test_X.reshape(len(ICUType2_ids), 48, -1)
    #ICUType_3_test_X = ICUType_3_test_X.reshape(len(ICUType_3_ids), 48, -1)
    #ICUType_4_test_X = ICUType_4_test_X.reshape(len(ICUType_4_ids), 48, -1)
    classificacao_undefined_test_X = classificacao_undefined_test_X.reshape(len(classificacao_undefined_ids), 48, -1)
    classificacao_baixo_peso_test_X = classificacao_baixo_peso_test_X.reshape(len(classificacao_baixo_peso_ids), 48, -1)
    classificacao_normal_peso_test_X = classificacao_normal_peso_test_X.reshape(len(classificacao_normal_peso_ids), 48, -1)
    classificacao_sobrepeso_test_X = classificacao_sobrepeso_test_X.reshape(len(classificacao_sobrepeso_ids), 48, -1)
    classificacao_obesidade_test_X = classificacao_obesidade_test_X.reshape(len(classificacao_obesidade_ids), 48, -1)
    #classificacao_obesidade_2_test_X = classificacao_obesidade_2_test_X.reshape(len(classificacao_obesidade_2_ids), 48, -1)
    #classificacao_obesidade_3_test_X = classificacao_obesidade_3_test_X.reshape(len(classificacao_obesidade_3_ids), 48, -1)

    
    
    # fetch labels for train/val/test sets
    train_y = y[y.index.isin(train_set_ids)].sort_index()
    val_y = y[y.index.isin(val_set_ids)].sort_index()
    test_y = y[y.index.isin(test_set_ids)].sort_index()
    female_gender_test_y = y[y.index.isin(female_gender_test_ids)].sort_index()
    male_gender_test_y = y[y.index.isin(male_gender_test_ids)].sort_index()
    undefined_gender_test_y = y[y.index.isin(undefined_gender_test_ids)].sort_index()
    more_than_or_equal_to_65_test_y = y[y.index.isin(more_than_or_equal_to_65_test_ids)].sort_index()
    less_than_65_test_y = y[y.index.isin(less_than_65_test_ids)].sort_index()
    #ICUType_1_test_y = y[y.index.isin(ICUType_1_ids)].sort_index()
    #ICUType_2_test_y = y[y.index.isin(ICUType2_ids)].sort_index()
    #ICUType_3_test_y = y[y.index.isin(ICUType_3_ids)].sort_index()
    #ICUType_4_test_y = y[y.index.isin(ICUType_4_ids)].sort_index()
    classificacao_undefined_test_y = y[y.index.isin(classificacao_undefined_ids)].sort_index()
    classificacao_baixo_peso_test_y = y[y.index.isin(classificacao_baixo_peso_ids)].sort_index()
    classificacao_normal_peso_test_y = y[y.index.isin(classificacao_normal_peso_ids)].sort_index()
    classificacao_sobrepeso_test_y = y[y.index.isin(classificacao_sobrepeso_ids)].sort_index()
    classificacao_obesidade_test_y = y[y.index.isin(classificacao_obesidade_ids)].sort_index()
    #classificacao_obesidade_2_test_y = y[y.index.isin(classificacao_obesidade_2_ids)].sort_index()
    #classificacao_obesidade_3_test_y = y[y.index.isin(classificacao_obesidade_3_ids)].sort_index()

    train_y = train_y.to_numpy()
    val_y = val_y.to_numpy() 
    test_y = test_y.to_numpy()
    female_gender_test_y = female_gender_test_y.to_numpy() 
    male_gender_test_y = male_gender_test_y.to_numpy()
    undefined_gender_test_y = undefined_gender_test_y.to_numpy()
    more_than_or_equal_to_65_test_y = more_than_or_equal_to_65_test_y.to_numpy()
    less_than_65_test_y = less_than_65_test_y.to_numpy()
    #ICUType_1_test_y = ICUType_1_test_y.to_numpy()
    #ICUType_2_test_y = ICUType_2_test_y.to_numpy()
    #ICUType_3_test_y = ICUType_3_test_y.to_numpy()
    #ICUType_4_test_y = ICUType_4_test_y.to_numpy()
    classificacao_undefined_test_y = classificacao_undefined_test_y.to_numpy()
    classificacao_baixo_peso_test_y = classificacao_baixo_peso_test_y.to_numpy()
    classificacao_normal_peso_test_y = classificacao_normal_peso_test_y.to_numpy()
    classificacao_sobrepeso_test_y = classificacao_sobrepeso_test_y.to_numpy()
    classificacao_obesidade_test_y = classificacao_obesidade_test_y.to_numpy()
    #classificacao_obesidade_2_test_y = classificacao_obesidade_2_test_y.to_numpy()
    #classificacao_obesidade_3_test_y = classificacao_obesidade_3_test_y.to_numpy()
   

    # fetch ICUType for train/val/test sets
    train_ICUType = ICUType[ICUType.index.isin(train_set_ids)].sort_index()
    val_ICUType = ICUType[ICUType.index.isin(val_set_ids)].sort_index()
    test_ICUType = ICUType[ICUType.index.isin(test_set_ids)].sort_index()
    test_ICUType_female_gender = ICUType[ICUType.index.isin(female_gender_test_ids)].sort_index()
    test_ICUType_male_gender = ICUType[ICUType.index.isin(male_gender_test_ids)].sort_index()
    test_ICUType_undefined_gender = ICUType[ICUType.index.isin(undefined_gender_test_ids)].sort_index()
    test_ICUType_more_than_or_equal_to_65 = ICUType[ICUType.index.isin(more_than_or_equal_to_65_test_ids)].sort_index()
    test_ICUType_less_than_65 = ICUType[ICUType.index.isin(less_than_65_test_ids)].sort_index()
    #test_ICUType_1 = ICUType[ICUType.index.isin(ICUType_1_ids)].sort_index()
    #test_ICUType_2 = ICUType[ICUType.index.isin(ICUType2_ids)].sort_index()
    #test_ICUType_3 = ICUType[ICUType.index.isin(ICUType_3_ids)].sort_index()
    #test_ICUType_4 = ICUType[ICUType.index.isin(ICUType_4_ids)].sort_index()
    test_ICUType_classificacao_undefined = ICUType[ICUType.index.isin(classificacao_undefined_ids)].sort_index()
    test_ICUType_classificao_baixo_peso = ICUType[ICUType.index.isin(classificacao_baixo_peso_ids)].sort_index()
    test_ICUType_classificacao_normal_peso = ICUType[ICUType.index.isin(classificacao_normal_peso_ids)].sort_index()
    test_ICUType_classificacao_sobrepeso = ICUType[ICUType.index.isin(classificacao_sobrepeso_ids)].sort_index()
    test_ICUType_classificacao_obesidade = ICUType[ICUType.index.isin(classificacao_obesidade_ids)].sort_index()
    #test_ICUType_classificacao_obesidade_2 = ICUType[ICUType.index.isin(classificacao_obesidade_2_ids)].sort_index()
    #test_ICUType_classificacao_obesidade_3 = ICUType[ICUType.index.isin(classificacao_obesidade_3_ids)].sort_index()

    train_ICUType, val_ICUType, test_ICUType, test_ICUType_female_gender, test_ICUType_male_gender, test_ICUType_undefined_gender,  test_ICUType_more_than_or_equal_to_65, test_ICUType_less_than_65, test_ICUType_classificacao_undefined, test_ICUType_classificao_baixo_peso, test_ICUType_classificacao_normal_peso, test_ICUType_classificacao_sobrepeso, test_ICUType_classificacao_obesidade= (
        train_ICUType.to_numpy(),
        val_ICUType.to_numpy(),
        test_ICUType.to_numpy(),
        test_ICUType_female_gender.to_numpy(),
        test_ICUType_male_gender.to_numpy(),
        test_ICUType_undefined_gender.to_numpy(),
        test_ICUType_more_than_or_equal_to_65.to_numpy(),
        test_ICUType_less_than_65.to_numpy(),
        #test_ICUType_1.to_numpy(),
        #test_ICUType_2.to_numpy(),
        #test_ICUType_3.to_numpy(),
        #test_ICUType_4.to_numpy(),
        test_ICUType_classificacao_undefined.to_numpy(),
        test_ICUType_classificao_baixo_peso.to_numpy(),
        test_ICUType_classificacao_normal_peso.to_numpy(),
        test_ICUType_classificacao_sobrepeso.to_numpy(),
        test_ICUType_classificacao_obesidade.to_numpy(),
        #test_ICUType_classificacao_obesidade_2.to_numpy(),
        #test_ICUType_classificacao_obesidade_3.to_numpy()
    )

    # assemble the final processed data into a dictionary
    processed_dataset = {
        # general info
        "n_classes": 2,
        "n_steps": 48,
        "n_features": train_X.shape[-1],
        "scaler": scaler,
        # train set
        "train_X": train_X,
        "train_y": train_y.flatten(),
        "train_ICUType": train_ICUType.flatten(),
        # val set
        "val_X": val_X,
        "val_y": val_y.flatten(),
        "val_ICUType": val_ICUType.flatten(),
        # test set
        "test_X": test_X,
        "test_y": test_y.flatten(),
        "test_ICUType": test_ICUType.flatten(),
        "female_gender_test_X":  female_gender_test_X,
        "female_gender_test_y":  female_gender_test_y.flatten(),
        "test_ICUType_female_gender": test_ICUType_female_gender.flatten(),
        "male_gender_test_X": male_gender_test_X,
        "male_gender_test_y": male_gender_test_y.flatten(),
        "test_ICUType_male_gender": test_ICUType_male_gender.flatten(),
        "undefined_gender_test_X": undefined_gender_test_X,
        "undefined_gender_test_y": undefined_gender_test_y.flatten(),
        "test_ICUType_undefined_gender": test_ICUType_undefined_gender.flatten(),
        "more_than_or_equal_to_65_test_X": more_than_or_equal_to_65_test_X,
        "more_than_or_equal_to_65_test_y": more_than_or_equal_to_65_test_y.flatten(),
        "test_ICUType_more_than_or_equal_to_65": test_ICUType_more_than_or_equal_to_65.flatten(),
        "less_than_65_test_X": less_than_65_test_X,
        "less_than_65_test_y": less_than_65_test_y.flatten(),
        "test_ICUType_less_than_65": test_ICUType_less_than_65.flatten(),
        #"ICUType_1_test_X" : ICUType_1_test_X,
        #"ICUType_1_test_y": ICUType_1_test_y.flatten(),
        #"test_ICUType_1": test_ICUType_1.flatten(),
        #"ICUType_2_test_X": ICUType_2_test_X,
        #"ICUType_2_test_y": ICUType_2_test_y.flatten(),
        #"test_ICUType_2": test_ICUType_2.flatten(),
        #"ICUType_3_test_X": ICUType_3_test_X,
        #"ICUType_3_test_y": ICUType_3_test_y.flatten(),
        #"test_ICUType_3": test_ICUType_4.flatten(),
        #"ICUType_4_test_X": ICUType_4_test_X,
        #"ICUType_4_test_y": ICUType_4_test_y.flatten(),
        #"test_ICUType_4": test_ICUType_4.flatten(),
        "classificacao_undefined_test_X": classificacao_undefined_test_X,
        "classificacao_undefined_test_y": classificacao_undefined_test_y.flatten(),
        "test_ICUType_classificacao_undefined": test_ICUType_classificacao_undefined.flatten(),
        "classificacao_baixo_peso_test_X": classificacao_baixo_peso_test_X,
        "classificacao_baixo_peso_test_y": classificacao_baixo_peso_test_y.flatten(),
        "test_ICUType_classificao_baixo_peso": test_ICUType_classificao_baixo_peso.flatten(),
        "classificacao_normal_peso_test_X": classificacao_normal_peso_test_X,
        "classificacao_normal_peso_test_y": classificacao_normal_peso_test_y.flatten(),
        "test_ICUType_classificacao_normal_peso": test_ICUType_classificacao_normal_peso.flatten(),
        "classificacao_sobrepeso_test_X": classificacao_sobrepeso_test_X,
        "classificacao_sobrepeso_test_y": classificacao_sobrepeso_test_y.flatten(),
        "test_ICUType_classificacao_sobrepeso": test_ICUType_classificacao_sobrepeso.flatten(),
        "classificacao_obesidade_test_X": classificacao_obesidade_test_X,
        "classificacao_obesidade_test_y": classificacao_obesidade_test_y.flatten(),
        "test_ICUType_classificacao_obesidade": test_ICUType_classificacao_obesidade.flatten(),
        #"classificacao_obesidade_2_test_X": classificacao_obesidade_2_test_X,
        #"classificacao_obesidade_2_test_y": classificacao_obesidade_2_test_y.flatten(),
        #"test_ICUType_classificacao_obesidade_2": test_ICUType_classificacao_obesidade_2.flatten(),
        #"classificacao_obesidade_3_test_X": classificacao_obesidade_3_test_X,
        #"classificacao_obesidade_3_test_y": classificacao_obesidade_3_test_y.flatten(),
        #"test_ICUType_classificacao_obesidade_3": test_ICUType_classificacao_obesidade_3.flatten(),
        #normalization
        "scaler":scaler
    }

    if rate > 0:
        logger.warning(
            "Note that physionet_2012 has sparse observations in the time series, "
            "hence we don't add additional missing values to the training dataset. "
        )

        # hold out ground truth in the original data for evaluation
        val_X_ori = val_X
        test_X_ori = test_X
        female_gender_test_X_ori = female_gender_test_X
        male_gender_test_X_ori = male_gender_test_X
        undefined_gender_test_X_ori = undefined_gender_test_X
        more_than_or_equal_to_65_test_X_ori = more_than_or_equal_to_65_test_X
        less_than_65_test_X_ori = less_than_65_test_X
        #ICUType_1_test_X_ori = ICUType_1_test_X 
        #ICUType_2_test_X_ori = ICUType_2_test_X
        #ICUType_3_test_X_ori = ICUType_3_test_X
        #ICUType_4_test_X_ori = ICUType_4_test_X
        classificacao_undefined_test_X_ori = classificacao_undefined_test_X
        classificacao_baixo_peso_test_X_ori = classificacao_baixo_peso_test_X
        classificacao_normal_peso_test_X_ori = classificacao_normal_peso_test_X
        classificacao_sobrepeso_test_X_ori = classificacao_sobrepeso_test_X
        classificacao_obesidade_test_X_ori = classificacao_obesidade_test_X
        #classificacao_obesidade_2_test_X_ori = classificacao_obesidade_2_test_X
        #classificacao_obesidade_3_test_X_ori = classificacao_obesidade_3_test_X

        # mask values in the validation set as ground truth
        val_X = create_missingness(val_X, rate, pattern, **kwargs)
        # mask values in the test set as ground truth
        test_X = create_missingness(test_X, rate, pattern, **kwargs)
        female_gender_test_X = create_missingness(female_gender_test_X, rate, pattern, **kwargs)
        male_gender_test_X = create_missingness(male_gender_test_X, rate, pattern, **kwargs)
        undefined_gender_test_X = create_missingness(undefined_gender_test_X, rate, pattern, **kwargs)
        more_than_or_equal_to_65_test_X = create_missingness(more_than_or_equal_to_65_test_X, rate, pattern, **kwargs)
        less_than_65_test_X = less_than_65_test_X = create_missingness(less_than_65_test_X, rate, pattern, **kwargs)
        #ICUType_1_test_X = create_missingness(ICUType_1_test_X, rate, pattern, **kwargs)
        #ICUType_2_test_X = create_missingness(ICUType_2_test_X, rate, pattern, **kwargs)
        #ICUType_3_test_X = create_missingness(ICUType_3_test_X, rate, pattern, **kwargs)
        #ICUType_4_test_X = create_missingness(ICUType_4_test_X, rate, pattern, **kwargs)
        classificacao_undefined_test_X = create_missingness(classificacao_undefined_test_X, rate, pattern, **kwargs)
        classificacao_baixo_peso_test_X = create_missingness(classificacao_baixo_peso_test_X, rate, pattern, **kwargs)
        classificacao_normal_peso_test_X = create_missingness(classificacao_normal_peso_test_X, rate, pattern, **kwargs)
        classificacao_sobrepeso_test_X = create_missingness(classificacao_sobrepeso_test_X, rate, pattern, **kwargs)
        classificacao_obesidade_test_X = create_missingness(classificacao_obesidade_test_X, rate, pattern, **kwargs)
        #classificacao_obesidade_2_test_X = create_missingness(classificacao_obesidade_2_test_X, rate, pattern, **kwargs)
        #classificacao_obesidade_3_test_X = create_missingness(classificacao_obesidade_3_test_X, rate, pattern, **kwargs)

        processed_dataset["train_X"] = train_X

        processed_dataset["val_X"] = val_X
        processed_dataset["val_X_ori"] = val_X_ori
        val_X_indicating_mask = np.isnan(val_X_ori) ^ np.isnan(val_X)
        logger.info(
            f"{val_X_indicating_mask.sum()} values masked out in the val set as ground truth, "
            f"take {val_X_indicating_mask.sum() / (~np.isnan(val_X_ori)).sum():.2%} of the original observed values"
        )

        processed_dataset["test_X"] = test_X
        processed_dataset["female_gender_test_X"] = female_gender_test_X
        processed_dataset["male_gender_test_X"] = male_gender_test_X
        processed_dataset["undefined_gender_test_X"] = undefined_gender_test_X
        processed_dataset["more_than_or_equal_to_65_test_X"] = more_than_or_equal_to_65_test_X 
        processed_dataset["less_than_65_test_X"] = less_than_65_test_X
        #processed_dataset["ICUType_1_test_X"] = ICUType_1_test_X
       # processed_dataset["ICUType_2_test_X"] =ICUType_2_test_X
        #processed_dataset["ICUType_3_test_X"] = ICUType_3_test_X
        #processed_dataset["ICUType_4_test_X"] = ICUType_4_test_X
        processed_dataset["classificacao_undefined_test_X"] = classificacao_undefined_test_X 
        processed_dataset["classificacao_baixo_peso_test_X"] = classificacao_baixo_peso_test_X
        processed_dataset["classificacao_normal_peso_test_X"] = classificacao_normal_peso_test_X
        processed_dataset["classificacao_sobrepeso_test_X"] = classificacao_sobrepeso_test_X
        processed_dataset["classificacao_obesidade_test_X"] = classificacao_obesidade_test_X
        #processed_dataset["classificacao_obesidade_2_test_X"] = classificacao_obesidade_2_test_X
        #processed_dataset["classificacao_obesidade_3_test_X"] = classificacao_obesidade_3_test_X
    
        processed_dataset["test_X_ori"] = test_X_ori
        processed_dataset["female_gender_test_X_ori"] = female_gender_test_X_ori
        processed_dataset["male_gender_test_X_ori"] = male_gender_test_X_ori
        processed_dataset["undefined_gender_test_X_ori"] = undefined_gender_test_X_ori
        processed_dataset["more_than_or_equal_to_65_test_X_ori"] = more_than_or_equal_to_65_test_X_ori 
        processed_dataset["less_than_65_test_X_ori"] = less_than_65_test_X_ori
        #processed_dataset["ICUType_1_test_X_ori"] = ICUType_1_test_X_ori
        #processed_dataset["ICUType_2_test_X_ori"] = ICUType_2_test_X_ori
        #processed_dataset["ICUType_3_test_X_ori"] = ICUType_3_test_X_ori
        #processed_dataset["ICUType_4_test_X_ori"] = ICUType_4_test_X_ori
        processed_dataset["classificacao_undefined_test_X_ori"] = classificacao_undefined_test_X_ori 
        processed_dataset["classificacao_baixo_peso_test_X_ori"] = classificacao_baixo_peso_test_X_ori
        processed_dataset["classificacao_normal_peso_test_X_ori"] = classificacao_normal_peso_test_X_ori
        processed_dataset["classificacao_sobrepeso_test_X_ori"] = classificacao_sobrepeso_test_X_ori
        processed_dataset["classificacao_obesidade_test_X_ori"] = classificacao_obesidade_test_X_ori
        #processed_dataset["classificacao_obesidade_2_test_X_ori"] = classificacao_obesidade_2_test_X_ori
        #processed_dataset["classificacao_obesidade_3_test_X_ori"] = classificacao_obesidade_3_test_X_ori

        test_X_indicating_mask = np.isnan(test_X_ori) ^ np.isnan(test_X)
        logger.info(
            f"{test_X_indicating_mask.sum()} values masked out in the test set as ground truth, "
            f"take {test_X_indicating_mask.sum() / (~np.isnan(test_X_ori)).sum():.2%} of the original observed values"
        )
    else:
        logger.warning("rate is 0, no missing values are artificially added.")

    print_final_dataset_info(train_X, val_X, test_X)
    return processed_dataset
