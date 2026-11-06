# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# %%
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

data = pd.read_csv(
    BASE_DIR / "data" / "sustainable_concrete_dataset.csv"
)

print(data.shape)
# %%
data.info()

# %%
data.describe()

# %%
data.head()

# %%
data.isnull().sum()

# %%
plt.scatter(data['compressive_strength'].values,data['energy_consumption'].values)
plt.xlabel("compressive_strength")
plt.ylabel("energy_consumption")
plt.title("Strength_v/s_Energy")
plt.show()

# %%
sorted_data = data.sort_values(by='compressive_strength', ascending=True)

subset = sorted_data.head(100)

plt.plot(subset['compressive_strength'],subset['energy_consumption'])
        
plt.xlabel("Compressive Strength")
plt.ylabel("Energy Consumption")
plt.title("Strength vs Energy (First 100 Sorted Points)")

plt.xlim(left=80)
plt.ylim(bottom=750)

plt.show()

# %%
X = data.drop(columns=['compressive_strength','energy_consumption']).values
Y = data[['compressive_strength','energy_consumption']].values

# %%
scaler = MinMaxScaler()

X = scaler.fit_transform(X)
Y = scaler.fit_transform(Y)

# %%
X_train, X_temp, Y_train, Y_temp = train_test_split(
    X, Y, test_size=0.3, random_state=1)

X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp, Y_temp, test_size=0.5, random_state=1)

# %%
model = Sequential()

model.add(Dense(8, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='linear'))


model.compile(
    optimizer='adam',
    loss='mse'
)

# %%
early = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=200,
    batch_size=16,
    callbacks=[early]
)

# %%
def evaluate(X,Y,name):

    pred = model.predict(X)

    mse = mean_squared_error(Y,pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y,pred)

    print(name)
    print("MSE:",mse)
    print("RMSE:",rmse)
    print("R2:",r2)

    return pred

# %%
train_pred = evaluate(X_train,Y_train,"TRAIN")
val_pred   = evaluate(X_val,Y_val,"VALIDATION")
test_pred  = evaluate(X_test,Y_test,"TEST")

# %%
model.summary()

# %%
plt.scatter(Y_test[:,0],test_pred[:,0])
plt.xlabel("True Strength")
plt.ylabel("Predicted Strength")
plt.title("Test: Compressive Strength")
plt.show()

plt.scatter(Y_test[:,1],test_pred[:,1])
plt.xlabel("True Energy")
plt.ylabel("Predicted Energy")
plt.title("Test: Energy Consumption")
plt.show()

plt.scatter(Y_val[:,0],val_pred[:,0])
plt.xlabel("True Strength")
plt.ylabel("Predicted Strength")
plt.title("Val: Compressive Strength")
plt.show()

plt.scatter(Y_val[:,1],val_pred[:,1])
plt.xlabel("True Energy")
plt.ylabel("Predicted Energy")
plt.title("Val: Energy Consumption")
plt.show()

plt.scatter(Y_train[:,0],train_pred[:,0])
plt.xlabel("True Strength")
plt.ylabel("Predicted Strength")
plt.title("Train: Compressive Strength")
plt.show()

plt.scatter(Y_train[:,1],train_pred[:,1])
plt.xlabel("True Energy")
plt.ylabel("Predicted Energy")
plt.title("Train: Energy Consumption")
plt.show()

# %%
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


class ConcreteProblem(Problem):

    def __init__(self):

        super().__init__(
            n_var=X.shape[1],
            n_obj=2,
            n_constr=0,
            xl=0,
            xu=1
        )

    def _evaluate(self,X_in,out,*args,**kwargs):

        pred = model.predict(X_in)

        strength = -pred[:,0]
        energy   = pred[:,1]

        out["F"] = np.column_stack([strength,energy])


algorithm = NSGA2(pop_size=100)


res = minimize(
    ConcreteProblem(),
    algorithm,
    ('n_gen',200),
    verbose=True
)


strength = -res.F[:,0]
energy = res.F[:,1]

plt.scatter(strength,energy)
plt.xlabel("Compressive Strength")
plt.ylabel("Energy Consumption")
plt.title("Pareto Front")
plt.show()


