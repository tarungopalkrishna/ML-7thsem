import pandas as pd
import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination


data = pd.read_csv("./data7.csv",names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg','thalach', 'exang', 'oldpeak', 'slope', 'ca','thal', 'heartdisease'])
data = data.replace("?",np.nan)
model = BayesianModel([('age', 'trestbps'), ('age', 'fbs'), ('sex','trestbps'),('exang','trestbps'),('trestbps','heartdisease'),('fbs','heartdisease'),('heartdisease','restecg'),('heartdisease','thalach'),('heartdisease','chol')])
model.fit(data,estimator=MaximumLikelihoodEstimator)
model_infr = VariableElimination(model)


ans = model_infr.query(variables=['heartdisease'],evidence={"age":28})

print(ans)