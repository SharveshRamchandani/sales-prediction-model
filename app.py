from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)

@app.route('/')
def index():
    # Load and process the dataset
    store_sales = pd.read_csv('train.csv')
    store_sales = store_sales.drop(['store', 'item'], axis=1)
    store_sales['date'] = pd.to_datetime(store_sales['date']).dt.to_period("M")
    monthly_sales = store_sales.groupby('date').sum().reset_index()
    monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()

    monthly_sales['sales_diff'] = monthly_sales['sales'].diff()
    monthly_sales = monthly_sales.dropna()

    supervised_data = monthly_sales.drop(['date', 'sales'], axis=1)
    for i in range(1, 13):
        supervised_data[f'month_{i}'] = supervised_data['sales_diff'].shift(i)

    supervised_data = supervised_data.dropna().reset_index(drop=True)
    train_data = supervised_data[:-12]
    test_data = supervised_data[-12:]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    X_train, y_train = train_data[:, 1:], train_data[:, 0:1].ravel()
    X_test, y_test = test_data[:, 1:], test_data[:, 0:1].ravel()

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pre = lr_model.predict(X_test).reshape(-1, 1)

    lr_pre_test_set = np.concatenate([lr_pre, X_test], axis=1)
    lr_pre_test_set = scaler.inverse_transform(lr_pre_test_set)

    result_list = []
    act_sales = monthly_sales['sales'][-13:].to_list()
    for index in range(len(lr_pre_test_set)):
        result_list.append(lr_pre_test_set[index][0] + act_sales[index])

    # Calculate metrics
    lr_mse = np.sqrt(mean_squared_error(result_list, monthly_sales['sales'][-12:]))
    lr_mae = mean_absolute_error(result_list, monthly_sales['sales'][-12:])
    lr_r2 = r2_score(result_list, monthly_sales['sales'][-12:])

    # Prepare predictions for rendering
    return render_template('index.html', 
                           actual_sales=monthly_sales['sales'].tolist()[-12:],
                           predicted_sales=result_list,
                           mse=lr_mse, mae=lr_mae, r2=lr_r2)

if __name__ == '__main__':
    app.run(debug=True)
