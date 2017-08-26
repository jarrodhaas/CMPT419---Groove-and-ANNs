# ETM modeller. Code from
#http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


dataset = numpy.loadtxt("training1.0.csv", delimiter=",")

# split into train/test data, but make labels for analysis, post-prediction
labeled_data = dataset[:,0:196]
np.random.shuffle(labeled_data)


with open('shuffled1.0.csv','w') as f_handle:
    np.savetxt(f_handle,labeled_data,fmt='%d', delimiter=",")

X_train = labeled_data[0:518,0:193]
Y_train = labeled_data[0:518,194:195]

X_test = labeled_data[518:,0:193]
Y_test = labeled_data[518:,194:195]

# define 10-fold cross validation test harness
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
#cvscores = []

# crossvalidation
#for train, test in kfold.split(X, Y):

# create model
model = Sequential()
model.add(Dense(193, input_dim=193, init='uniform', activation='relu'))
model.add(Dense(150, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, Y_train, nb_epoch=150, batch_size=100)

# evaluate the model
scores = model.evaluate(X_test, Y_test)
print("\n\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#for cross validation
#cvscores.append(scores[1] * 100)

#print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))

# calculate predictions
#predictions = model.predict(X)
# round predictions
#rounded = [round(x) for x in predictions]
#print(rounded)
