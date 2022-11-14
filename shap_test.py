#!/bin/env python3
import shap
import xgboost
import sklearn
from matplotlib import pyplot as plt

X,y = shap.datasets.adult()
X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=42)
model = xgboost.XGBClassifier().fit(X_train.values, y_train)

# workaround for SHAP bug - see
# see https://github.com/slundberg/shap/issues/1215
if True:
    mybooster = model.get_booster()

    model_bytearray = mybooster.save_raw()[4:]
    def myfun(self=None):
        return model_bytearray

    mybooster.save_raw = myfun
    model = mybooster

explainer = shap.explainers.Tree(model, feature_names=X_valid.columns)
shap_values = explainer(X_valid.iloc[0:1000,:].values)

plt.figure(figsize=(10,15))
shap.plots.beeswarm(shap_values)
plt.tight_layout()
plt.savefig('shaptest.png')
