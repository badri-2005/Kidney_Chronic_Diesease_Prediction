# 🩺 Kidney Disease Prediction using Machine Learning

This project is a supervised machine learning solution designed to predict the presence of chronic kidney disease (CKD) based on various medical features. The model is trained and tested on a real-world dataset using different classification algorithms to compare their effectiveness.

---

## 📊 Dataset Description - Kaggle Dataset 

The dataset `kidney_disease.csv` contains various medical parameters such as:

- Age
- Blood pressure
- Specific gravity
- Albumin
- Sugar
- Red blood cells
- Pus cell count
- Serum creatinine
- Sodium, Potassium levels
- Hemoglobin
- Packed cell volume
- White blood cell count
- And more...

The target variable is `classification`, indicating whether the patient has chronic kidney disease.

---

## 🛠️ Technologies and Libraries Used

- Python 3.x  
- NumPy  
- Pandas  
- scikit-learn  
- Matplotlib  
- Seaborn  

---

## ⚙️ Project Workflow

1. **Import Libraries**
   - Load necessary Python libraries for data analysis, visualization, and machine learning.

2. **Load and Explore Dataset**
   - Read the CSV file and perform exploratory data analysis (EDA).
   - Check data types, missing values, and visualize data distributions.

3. **Data Cleaning and Preprocessing**
   - Handle missing values using `SimpleImputer` (mean strategy for numerical features).
   - Encode categorical variables as required.
   - Split the dataset into features (`X`) and target (`Y`).
   - Standardize/scale the feature set (if needed).

4. **Train-Test Split**
   - Split data into `train` and `test` sets using `train_test_split` from scikit-learn.

5. **Model Selection**
   - Create a list of various machine learning classification models:
     - Logistic Regression
     - Decision Tree Classifier
     - Random Forest Classifier
     - Support Vector Classifier (SVC)
     - Gaussian Naive Bayes

6. **Model Training & Evaluation**
   - Loop through each model:
     - Fit the model on training data.
     - Predict on test data.
     - Evaluate using:
       - Confusion Matrix
       - Accuracy Score
       - Classification Report
   - Handle missing values in training and test sets using:
     ```python
     from sklearn.impute import SimpleImputer
     imputer = SimpleImputer(strategy='mean')
     x_train = imputer.fit_transform(x_train)
     X_test = imputer.transform(X_test)
     ```

7. **Result Comparison**
   - Compare the accuracy and performance metrics of all models to identify the best performing algorithm.

---

## 📈 Results

- Multiple models were compared based on accuracy and classification reports.
- Insights were derived on which model performs better for this particular dataset.
- Example metrics used:
  - Accuracy Score
  - Confusion Matrix
  - Precision, Recall, F1-Score (from the classification report)

---

## 📌 Conclusion

This project demonstrates how machine learning can effectively predict chronic kidney disease based on patient health parameters. The process included:
- Cleaning and preprocessing real-world healthcare data
- Handling missing values using imputation techniques
- Training and evaluating multiple ML models
- Selecting the best model based on performance metrics

---


## 📝 Author

**Badri Narayanan**  
*B.E. Computer Science Engineering Student | ML Enthusiast | Data Analyst*


---

