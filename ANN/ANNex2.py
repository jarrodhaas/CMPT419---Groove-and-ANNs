# ETM modeller. Code from
#http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


dataset = np.loadtxt("training~LC.csv", delimiter=",")

# split into train/test data, but make labels for analysis, post-prediction
X = dataset[:,0:193]
Y = dataset[:,193]


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)



def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(193, input_dim=193, init='normal', activation='relu'))
	model.add(Dense(600, init='uniform', activation='relu'))

	model.add(Dense(4, init='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# Fit the model
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=700, batch_size=200, verbose=1)

# evaluate the model

kfold = KFold(n_splits=2, shuffle=True, random_state=seed)

results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
