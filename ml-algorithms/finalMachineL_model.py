import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)
def read_data():
    dataset = pd.read_csv('outfinal.csv',delimiter=',')
    dataset = dataset.drop(['rating'],axis = 1)
    dataset = clean_dataset(dataset)
    return dataset
def graphs():
    df = read_data()
    test = df['succesful']
    train = df.drop(['succesful'],axis=1)
    print("Train dataset has {} samples and {} attributes".format(*train.shape))
    print("Test dataset has {} samples and ".format(*test.shape))
    fig , ax = plt.subplots(figsize=(6,4))
    train.adc1_740_mean[train.adc1_740_mean > 3000] = train.adc1_740_mean.mean()
    train.adc2_740_mean[train.adc2_740_mean > 1500] = train.adc2_740_mean.mean()
    train.adc3_740_mean[train.adc3_740_mean > 500] = train.adc3_740_mean.mean()
    train.adc4_740_mean[train.adc4_740_mean > 300] = train.adc4_740_mean.mean()
    train.adc1_850_mean[train.adc1_850_mean > 3000] = train.adc1_850_mean.mean()
    train.adc2_740_mean[train.adc2_740_mean > 1500] = train.adc2_850_mean.mean()
    train.adc3_740_mean[train.adc3_740_mean > 800] = train.adc3_850_mean.mean()
    train.adc4_740_mean[train.adc4_740_mean > 400] = train.adc4_850_mean.mean()
    scaler = MinMaxScaler()
    features = [['adc1_740_std','adc2_740_std','adc3_740_std','adc4_740_std',
            'adc1_850_std','adc2_850_std','adc3_850_std','adc4_850_std','pulse_mean','pulse_std']]

    for feature in features:
        train[feature] = scaler.fit_transform(train[feature])
    sns.countplot(x='succesful', data=df)
    plt.title("Count of succesful and unsuccesful")
    plt.show()
    print(train.head())
    corr_df=train  #New dataframe to calculate correlation between numeric features
    cor= corr_df.corr(method='pearson')
    print(cor)
    fig, ax =plt.subplots(figsize=(8, 6))
    plt.title("Correlation Plot")
    sns.heatmap(cor, mask=np.zeros_like(cor, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax)
    plt.show()
    print(train.isnull().sum())
    print(train.describe())
    x_train, x_test, y_train, y_test = train_test_split(train, test, test_size=0.25, random_state=4)
    rfc=KNeighborsClassifier(n_neighbors=16)
    rfc.fit(x_train,y_train)
    
    y_pred = rfc.predict(x_test)
    y_true=y_test
    cm = confusion_matrix(y_true, y_pred)

    f, ax =plt.subplots(figsize = (5,5))

    sns.heatmap(cm,annot = True, linewidths= 0.5, linecolor="red", fmt=".0f", ax=ax)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    plt.show()
    print("Test set accuracy: {:.2f}".format(rfc.score(x_test, y_test)))
    print(y_pred)

import os



graphs()
