# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# ## Objective - Create Simple Neural Network
# %% [markdown]
# #### import modules and prepare data

# %%
# Import sklearn/tensorflow modules.
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create simple dataset.
X_train = np.arange(-2, 10, dtype=float)
y_train = ((X_train * 2) - 1).astype(float)

# The rule is y = 2x - 1

print(X_train, X_train.dtype)
print(y_train, y_train.dtype)

# %% [markdown]
# #### Create model

# %%
def model():
    model = Sequential()
    model.add(Dense(units=1, input_shape=[1]))
    model.compile(optimizer='sgd', loss='mean_squared_error')
    return model

model = model()

results = model.fit(X_train, y_train, epochs=150, verbose=False)    # NOTE: epochs are increased because of limited data.

print('Loss: {:.2f}'.format(model.evaluate(X_train, y_train, verbose=False)))

# %% [markdown]
# #### Test model

# %%
def test():
    for i in range(4):
        random_num = np.random.randint(-4, 20)
        pred = model.predict([random_num])[0][0]

        if round(pred) == random_num * 2 - 1:
            output = '✔️'
        else:
            output = '❌'

        print(np.array([random_num, round(pred)]), output)

test()

# %% [markdown]
# #### Plot the results

# %%
fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(10, 5))

slope = model.get_weights()[0][0]
intercept = model.get_weights()[1]

ax_1.plot(X_train, X_train * slope + intercept)
ax_1.scatter(X_train, y_train)

slope = '{:.2}'.format(slope[0])
intercept = '{:.2}'.format(intercept[0])

print('{:^4} {:^4}'.format(slope, intercept))
print('{:^4} {:^4}'.format(2, -1))

ax_2.plot(results.history['loss'])
ax_2.set_xlabel('epoch')
ax_2.set_ylabel('loss')


