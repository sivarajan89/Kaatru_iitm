import pandas as pd
from sklearn import linear_model as lm
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Linear Regression Model

df = pd.read_csv('day.csv')
n = len(df)
print('The length of the database is ' + str(n))

x = df[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']]
y = df['cnt']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
r = lm.LinearRegression()
r.fit(x_train, y_train)

print(r.coef_)

y_pred = r.predict(x_test)

acc = r2_score(y_test, y_pred) * 100
print('The Accuracy of Linear Regression Model is ' + str(acc))

params = np.append(r.intercept_, r.coef_)

new_x = np.append(np.ones((len(x_test), 1)), x_test, axis=1)
mse = (sum((y_test-y_pred)**2))/(len(new_x)-len(new_x[0]))
v_b = mse*(np.linalg.inv(np.dot(new_x.T, new_x)).diagonal())
s_b = np.sqrt(v_b)
t_b = params/s_b

p_val = [2*(1-stats.t.cdf(np.abs(i),(len(new_x)-len(new_x[0])))) for i in t_b]
p_val = np.round(p_val, 3)
print(['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed'])
print('The p value of Linear Regression Model is ' + str(p_val))
