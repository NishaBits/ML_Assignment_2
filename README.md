a) Problem Statement
The objective of this project is to build and compare multiple machine learning classification models to predict whether a breast tumor is benign (0) or malignant (1) based on medical diagnostic features. Models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

b) Dataset Description
We used the Breast Cancer Wisconsin dataset from scikit-learn (originally from the UCI Machine Learning Repository). The dataset contains 569 instances and 30 numeric features. The target variable is binary: 0 = benign, 1 = malignant. This satisfies the assignment requirements of at least 12 features and 500 instances.

c) Models Used and Evaluation Metrics
ML MODEL NAME	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic
Regression	0.956140	0.997707	0.945946	0.985915	0.965517	0.906811
Decision Tree	0.947368	0.943990	0.957746	0.957746	0.957746	0.887979
kNN	0.956140	0.995906	0.934211	1.000000	0.965986	0.908615
Naive Bayes	0.973684	0.998362	0.959459	1.000000	0.979310	0.944733
Random Forest (Ensemble)	0.964912	0.995251	0.958904	0.985915	0.972222	0.925285
XGBoost (Ensemble)	0.956140	0.995087	0.958333	0.971831	0.965035	0.906379
<img width="468" height="172" alt="image" src="https://github.com/user-attachments/assets/c92daee0-4023-4305-9d78-5d0af65f8a2c" />

d) Observations on Model Performance
ML Model Name	Observation about model performance
Logistic Regression	Performed very well with high accuracy (0.9561) and excellent AUC (0.9977), showing that the data is largely linearly separable. Stable model but slightly lower recall than some other models.
Decision Tree	Easy to interpret and reasonably accurate (0.9474), but slightly less stable than ensemble methods, indicating possible overfitting to training data.
kNN	Achieved perfect recall (1.0), meaning it correctly detected all malignant cases, but relies heavily on distance measures and can be sensitive to feature scaling.
Naive Bayes	Best overall performer with highest accuracy (0.9737), AUC (0.9984), F1 (0.9793), and MCC (0.9447), showing strong and balanced classification ability.
Random Forest (Ensemble)	Very stable and reliable model with consistently high performance across all metrics, reducing overfitting compared to a single decision tree.
XGBoost (Ensemble)	Strong ensemble model with good generalization, but slightly weaker than Random Forest and Naive Bayes on this dataset.

<img width="432" height="390" alt="image" src="https://github.com/user-attachments/assets/ba2257b8-d41e-47ed-b896-8c72f3fbf3d7" />
