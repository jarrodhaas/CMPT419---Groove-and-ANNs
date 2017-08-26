# ETM modeller. Code from
#http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


dataset = np.loadtxt("training~L~C~01R.csv", delimiter=",")

np.random.shuffle(dataset)

#predict 'real' groove
real_set = np.loadtxt("single.csv", delimiter=",")
X_real = real_set[:,0:193]

# split into train/test data, but make labels for analysis, post-prediction



X = dataset[:,0:193]
Y = dataset[:,193]

# define 10-fold cross validation test harness
fold_size = 2

kfold = StratifiedKFold(n_splits=fold_size, shuffle=True, random_state=seed)
cvscores = []

results = np.zeros((fold_size,2))
fold_num = 0

for train, test in kfold.split(X, Y):

    # create model
    model = Sequential()
    model.add(Dense(193, input_dim=193, init='uniform', activation='relu'))
    model.add(Dense(210, init='uniform', activation='relu'))
    model.add(Dense(210, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(X, Y, nb_epoch=150, batch_size=370)

    # evaluate the model
    scores = model.evaluate(X, Y)
    cvscores.append(scores[1] * 100)
    print("\n\n%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


    # calculate predictions
    pred = model.predict(X_real)

    print(pred)

    rpred = np.round(pred,0)

    corr0 = 0
    corr1 = 0

    for i in range(0,pred.size):
        if rpred[i] == Y[i]:
            if(Y[i]==0):
              corr0 +=1
            elif(Y[i]==1):
              corr1 +=1

    print corr0
    print corr1

    results[fold_num,0]=corr0
    results[fold_num,1]=corr1

    predh = 0

    for i in range(0,pred.size):
        if pred[i] > 0.51:
            predh+=1

    print (predh)

    #increment fold count
    fold_num +=1
    print (fold_num)

print results

print np.sum(results[:,0]) / (740*fold_size)
print np.sum(results[:,1]) / (185*fold_size)
