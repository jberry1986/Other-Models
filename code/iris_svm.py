import pandas as pd
import numpy as np
from sklearn import metrics,svm, pipeline, preprocessing, datasets
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
 
 
def get_trained_model(model_data):
    X_train, X_test, y_train, y_test = train_test_split(
                                           model_data.data, model_data.target,
                                           test_size=0.2)
    scalar = preprocessing.StandardScaler()
    clf = svm.SVC()
 
    pipeline = Pipeline([("scalar", scalar), ("classifier", clf)])
 
    kf = KFold(n_splits=5)
 
    param_grid = {"classifier__C": np.logspace(-4,4,9),
                  "classifier__gamma": np.logspace(-4,4,9)}
 
    gs = GridSearchCV(pipeline, param_grid, scoring="accuracy", cv=kf)
 
    gs.fit(X_train, y_train)
 
    test_accuracy = metrics.accuracy_score(gs.predict(X_test), y_test)
 
    return gs.best_estimator_ , test_accuracy
 
 
if __name__ == "__main__":
    print("Loading training data")
    iris_data = datasets.load_iris()
    print("Headers: {}".format(iris_data.feature_names))
    print("Training Model")
    trained_model, accuracy =  get_trained_model(iris_data)
    print("Test Set Model Accuracy")
    print(accuracy)
    joblib.dump(trained_model, "iris_model.pkl", compress=True)
    print(trained_model.get_params())