import pandas as pd
import numpy as np
from time import time
from sklearn.preprocessing import MinMaxScaler

def date_claimed_to_fraud(date):
    return 0.0 if pd.isna(date) else 1.0

def preprocess(train_file, test_file, fraud_file):
    t0 = time()
    # import files
    fraud_cases = pd.read_csv(fraud_file)
    CH_train = pd.read_csv(train_file)
    CH_test = pd.read_csv(test_file)
    
    train_len = len(CH_train)
    test_len = len(CH_test)
    
    # drop useless columns and possible bias columns
    drop_cols = ['sys_sector', 'sys_label', 'sys_process', 'sys_product',
                 'sys_dataspecification_version', 'sys_currency_code',
                 'ph_gender']
    CH_train = CH_train.drop(columns=drop_cols)
    CH_test = CH_test.drop(columns=drop_cols)
    
    # change claim ID format
    CH_train["sys_claimid"] = CH_train["sys_claimid"].map(lambda x: x[4:-3])
    CH_test["sys_claimid"] = CH_test["sys_claimid"].map(lambda x: x[4:-3])
    fraud_cases["ClaimID"] = fraud_cases["ClaimID"].map(lambda x: x[:9])
    
    # merge and convert 
    CH_train = pd.merge(CH_train, fraud_cases,
               left_on=["sys_claimid", "claim_date_occurred"],
               right_on=["ClaimID", "Date_Occurred"],
               how="outer")

    CH_train["sys_fraud"] = CH_train["Date_Occurred"].map(date_claimed_to_fraud)
    CH_train = CH_train.drop(columns=["ClaimID", "Date_Occurred"])
    
    # convert to datetime and create interval column
    CH_train["claim_date_reported"] = pd.to_datetime(CH_train["claim_date_reported"].astype(str))
    CH_train["claim_date_occurred"] = pd.to_datetime(CH_train["claim_date_occurred"].astype(str))

    CH_test["claim_date_reported"] = pd.to_datetime(CH_test["claim_date_reported"].astype(str))
    CH_test["claim_date_occurred"] = pd.to_datetime(CH_test["claim_date_occurred"].astype(str))

    CH_train["claim_time_interval"] = CH_train["claim_date_reported"] - CH_train["claim_date_occurred"] 
    CH_train["claim_time_interval"] = CH_train["claim_time_interval"].map(lambda x: x.days)

    CH_test["claim_time_interval"] = CH_test["claim_date_reported"] - CH_train["claim_date_occurred"] 
    CH_test["claim_time_interval"] = CH_test["claim_time_interval"].map(lambda x: x.days)
    
    # drop entries with incorrect reported dates
    CH_test = CH_test[CH_test["claim_date_reported"]<pd.to_datetime("2018-07-01")]
    CH_train = CH_train[CH_train["claim_date_reported"]<pd.to_datetime("2018-07-01")]
    
    # drop entries with missing causetypes
    CH_train = CH_train.dropna(subset=['claim_causetype'])
    CH_test = CH_test.dropna(subset=['claim_causetype'])
    
#     # substitute NaNs in policy_insured_amount with mean 
     
#     CH_train['policy_insured_amount'] = CH_train['policy_insured_amount'].fillna(
#                                                   CH_train['policy_insured_amount'].mean())

#     CH_test['policy_insured_amount'] = CH_test['policy_insured_amount'].fillna(
#                                                   CH_test['policy_insured_amount'].mean())
    
    # dummy columns for categorical features
    CH_train = pd.concat([CH_train.drop('claim_causetype', axis=1),
                    pd.get_dummies(CH_train['claim_causetype'], prefix='cause')], axis=1)
    CH_train = pd.concat([CH_train.drop('object_make', axis=1),
                    pd.get_dummies(CH_train['object_make'], prefix='make')], axis=1)
    CH_train = pd.concat([CH_train.drop('policy_profitability', axis=1),
                    pd.get_dummies(CH_train['policy_profitability'], prefix='profitability')], axis=1)

    CH_test = pd.concat([CH_test.drop('claim_causetype', axis=1),
                    pd.get_dummies(CH_test['claim_causetype'], prefix='cause')], axis=1)
    CH_test = pd.concat([CH_test.drop('object_make', axis=1),
                    pd.get_dummies(CH_test['object_make'], prefix='make')], axis=1)
    CH_test = pd.concat([CH_test.drop('policy_profitability', axis=1),
                    pd.get_dummies(CH_test['policy_profitability'], prefix='profitability')], axis=1)
    
    
    # date decomposition
    CH_train['occurred_year'] = CH_train["claim_date_occurred"].dt.year
    CH_train['occurred_month'] = CH_train["claim_date_occurred"].dt.month
    CH_train['occurred_day'] = CH_train["claim_date_occurred"].dt.day

    CH_train['reported_year'] = CH_train["claim_date_reported"].dt.year
    CH_train['reported_month'] = CH_train["claim_date_reported"].dt.month
    CH_train['reported_day'] = CH_train["claim_date_reported"].dt.day

    CH_test['occurred_year'] = CH_test["claim_date_occurred"].dt.year
    CH_test['occurred_month'] = CH_test["claim_date_occurred"].dt.month
    CH_test['occurred_day'] = CH_test["claim_date_occurred"].dt.day

    CH_test['reported_year'] = CH_test["claim_date_reported"].dt.year
    CH_test['reported_month'] = CH_test["claim_date_reported"].dt.month
    CH_test['reported_day'] = CH_test["claim_date_reported"].dt.day
    
    # Previous claims
    CH_train["full_name"] = CH_train["ph_name"] + " " + CH_train["ph_firstname"]
    CH_test["full_name"] = CH_test["ph_name"] + " " + CH_test["ph_firstname"]
    
    CH_train["set"] = ["train"]*len(CH_train)
    CH_test["set"] = ["test"]*len(CH_test)
    full_set = CH_train.append(CH_test)
    
    full_set = full_set.sort_values('claim_date_reported')
    full_set['prev_claims']=full_set.groupby('full_name')['full_name'].cumcount()
    
    CH_train = full_set[full_set["set"] == "train"]
    CH_test = full_set[full_set["set"] == "test"]

    drop_cols = ["ph_name", "ph_firstname", "full_name", "set", "claim_date_occurred", 
                 "claim_date_reported",'policy_insured_amount']
    CH_train = CH_train.drop(columns = drop_cols)
    CH_test = CH_test.drop(columns = drop_cols)
    
    # Normalization
    norm_cols = ['claim_amount_claimed_total', 'object_year_construction',  
             'claim_time_interval', 'occurred_year', 'occurred_month', 'occurred_day',
             'reported_year', 'reported_month', 'reported_day', 'prev_claims']

    for column in norm_cols:
        x = np.append(CH_train[column], CH_test[column])
        scaler = MinMaxScaler()
        scaler.fit(x.reshape(-1,1))
        CH_train[column] = scaler.transform(CH_train[[column]].values)
        CH_test[column] = scaler.transform(CH_test[[column]].values)

    assert (CH_train.columns == CH_test.columns).all()
    
    print("Finished preprocessing.")
    print(f"Dropped {train_len - len(CH_train)} Train entries.")
    print(f"Dropped {test_len - len(CH_test)} Test entries.")
    print("Preprocessing took {0:.2f} seconds".format(time()-t0))

    return CH_train, CH_test
    
    
    
    