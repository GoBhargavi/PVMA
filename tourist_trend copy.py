import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
import keras
from keras import losses, Model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import EarlyStopping

from keras.models import Sequential

scaler1, scaler2 = None,None
n_lookback, n_forecast = None, None

def pre_processing(visitor_path1,climate_path2,bangkok_path3, n_forecastx=10):

    global scaler1, scaler2, n_forecast

    n_forecast = n_forecastx

    df1 = pd.read_csv(visitor_path1)
    # print(df1)
    # input()
    df2 = pd.read_csv(climate_path2)
    # print(df2)
    # input()
    df3 = pd.read_csv(bangkok_path3)
    # print(df3)
    # input()
    # df = pd.concat([df1,df2,df3],ignore_index=True)
    df4 = pd.merge(df2,df3, how="outer", on=["date"])
    df = pd.merge(df1,df4, how="outer", on=["date"])
    # df = df.dropna()
    # print(df)
    # input()

    date = list(df["date"])
    month = []
    year = []
    for i in date:
        j = str(i).split('-')
        month.append(j[1])
        year.append(j[0])

    df["Months"] = month
    df["year"] = year
    df = df.drop(columns=["date"])

    # print(df.head(12))
    # input()


    done_date = set()
    for m,y in zip(df['Months'].values, df['year'].values):
        if (m,y) in done_date:
            # print(df[(df['year']== y) & (df['Months']== m)].index[-1])
            df = df.drop(df[(df['year']== y) & (df['Months']== m)].index[:-1])
        else:
            done_date.add((m,y))

    df.to_csv('final_dataset.csv', index=False)

    col = df.pop("visitors")
    df.insert(len(df.columns),col.name, col)
    col = df.pop("year")
    df.insert(1,col.name, col)

    # print(df.head(12))
    # input()

    y = df['visitors'].fillna(method='ffill')
    y = y.values.reshape(-1, 1)

    scaler1 = MinMaxScaler(feature_range=(0, 1))
    scaler1 = scaler1.fit(y)
    y = scaler1.transform(y)

    col = df.pop("tavg")
    df.insert(len(df.columns),col.name, col)
    col = df.pop("year")
    df.insert(1,col.name, col)

    t = df['tavg'].fillna(method='ffill')
    t = t.values.reshape(-1, 1)

    scaler2 = MinMaxScaler(feature_range=(0, 1))
    scaler2 = scaler2.fit(t)
    t = scaler2.transform(t)
    print(y,t)

    return y,t


def model(y,t):
    global n_lookback, n_forecast

    n_lookback = n_forecast * 2
   
   
    print(n_lookback, n_forecast)

    X = []
    Y = []

    got_data = 0

    for i in range(n_lookback, len(y) - n_forecast + 1):
        l1 = y[i - n_lookback: i]
        l2 = t[i - n_lookback: i]

        nlist = []

        
    
        for v in range(len(l1)):
            nlist.append(l1[v])
            nlist.append(l2[v])
        
        got_data = 1
    
    if got_data == 0:
        print("Reduce n_forecast")
        return
    
    # print(nlist)
    # print("#########")
    X.append(nlist)
    Y.append(y[i: i + n_forecast])

    X = np.array(X)
    Y = np.array(Y)

    if X.shape[0] == 0:
        print("Reduce n_forecast")
        return

    total_features_used = 2 

    n_lookback = n_lookback * total_features_used 
    # fit the model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(n_forecast))

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X, Y, epochs=100, batch_size=32, verbose=1)

    n_lookback = n_lookback//total_features_used

    X1 = y[- n_lookback:]
    X2 = t[- n_lookback:]

    X_ = []
    for i in range(len(X1)):
        X_.append(X1[i])
        X_.append(X2[i])

    print(X1.shape)
    X_ = np.array(np.array(X_))
    print("X_ Shape:", X_.shape)
    X_ = X_.reshape(1, n_lookback*total_features_used, 1)

    Y_ = model.predict(X_).reshape(-1, 1)

    # t = scaler.transform(t)
    Y_ = scaler1.inverse_transform(Y_)

    print("Y_ Shape:", Y_.shape)

    return X_,Y_



def forecast(Y_, filepath):
    global n_lookback, n_forecast

    df = pd.read_csv("final_dataset.csv")
    df_past = df[['visitors']].reset_index()
    df_past['Date'] = df['year'].astype(str) + "-" + df['Months'].astype(str) + "-15"

    df_past.rename(columns={'visitors': 'visitors_Actual'}, inplace=True)

    df_past['Date'] = df_past['Date']
    df_past['Date'] = pd.to_datetime(df_past['Date'])

    df_past['Forecast'] = np.nan
    df_past['Forecast'].iloc[-1] = df_past['visitors_Actual'].iloc[-1]

    df_future = pd.DataFrame(columns=['Date', 'visitors_Actual', 'Forecast'])
    df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=31), periods=n_forecast, freq='MS')
    df_future['Forecast'] = Y_.flatten()
    df_future['visitors_Actual'] = np.nan  

    results = df_past.append(df_future).set_index('Date')

    results.drop(columns=['index'], inplace=True)
    plot = results.plot(title='Visitor')
    fig = plot.get_figure()
    fig.savefig(f"{filepath}.png")


    results2 = results.tail(n_lookback)
    plot2 = results2.plot(title='Visitor')
    fig2 = plot2.get_figure()
    fig2.savefig(f"{filepath}_2.png")
   
    return results

def forecast(Y_, filepath):
    global n_lookback, n_forecast

    df = pd.read_csv("final_dataset.csv")
    df_past = df[['visitors']].reset_index()
    df_past['Date'] = df['day'].astype(str) + "-" + df['Months'].astype(str) + "-" + df['year'].astype(str)

    df_past.rename(columns={'visitors': 'visitors_Actual'}, inplace=True)

    df_past['Date'] = df_past['Date']
    df_past['Date'] = pd.to_datetime(df_past['Date'],dayfirst=True)

    df_past['Forecast'] = 0
    df_past['Forecast'].iloc[-1] = df_past['visitors_Actual'].iloc[-1]

    df_future = pd.DataFrame(columns=['Date', 'visitors_Actual', 'Forecast'])
    df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=7), periods=n_forecast, freq=pd.Timedelta(days=7))
    df_future['Forecast'] = Y_.flatten()
    df_future['visitors_Actual'] = 0 

    results = pd.concat([df_past, df_future], ignore_index=True).set_index('Date')

    results.drop(columns=['index'], inplace=True)
    plot = results.plot(title='Visitor')
    fig = plot.get_figure()
    
    fig.savefig(f"{filepath}.png")


    results2 = results.tail(n_lookback)
    plot2 = results2.plot(title='Visitor')
    fig2 = plot2.get_figure()
    fig2.savefig(f"{filepath}_2.png")
    
    # saving predicted visitors
    results.to_csv(f"{filepath}.csv")
   
    return results



def main(stat_data_path, climate_data_path, google_trends_data_path, filepath):
    y,t = pre_processing(stat_data_path, climate_data_path, google_trends_data_path, n_forecastx=4)
    resulting = model(y,t)
    final = forecast(resulting[1], filepath)
    return final

if __name__ == "__main__":
    main("./sample_data/statistics.csv","./sample_data/cilmate.csv","./sample_data/goole_trends.csv","./sample_data/test")
    
