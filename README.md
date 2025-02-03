# Credit-Card-Fraud-Detection

# 🏦 Credit Card Fraud Detection

This project focuses on **detecting fraudulent credit card transactions** using **Machine Learning** models. We compare the performance of **Random Forest** and **XGBoost** classifiers to identify fraud efficiently.

![Fraud Detection Banner](https://user-images.githubusercontent.com/your-image-path/banner.png)  

---

## 🚀 **Project Overview**
Credit card fraud is a growing concern in financial sectors. This project aims to develop a **machine learning-based fraud detection system** that can accurately classify transactions as **fraudulent (1) or genuine (0).**

### 🔥 **Key Features**
✅ **Data Preprocessing:** Handling imbalanced data, missing values, and feature selection.  
✅ **Model Training:** Uses **Random Forest** and **XGBoost** for classification.  
✅ **Cross-Validation:** Ensures models generalize well using **Stratified K-Fold CV**.  
✅ **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, and Confusion Matrix.  

---

## 💂 **Dataset**
We use the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) which contains **284,807 transactions**, with only **0.17% fraudulent cases**. The dataset is highly **imbalanced**, requiring special techniques for handling fraud detection.

- **Total Transactions:** 284,807  
- **Non-Fraud Cases (Class 0):** 284,315  
- **Fraud Cases (Class 1):** 492  

---

## 🛠 **Installation & Setup**
### **1⃣ Clone the Repository**
```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### **2⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3⃣ Run the Jupyter Notebook**
```bash
jupyter notebook
```

---

## 📊 **Machine Learning Models Used**
### 🔶 **1. Random Forest**
A **bagging-based ensemble method** that improves accuracy by training multiple decision trees.

### 🔶 **2. XGBoost**
A powerful **gradient boosting algorithm** optimized for large datasets and handling imbalanced data.

---

## 🏢 **Project Workflow**
1️⃣ **Data Preprocessing:**  
   - Checking missing values  
   - Feature scaling  
   - Handling class imbalance (e.g., SMOTE)  

2️⃣ **Train-Test Split:**  
   - **80% training data, 20% test data**  
   - Using `stratify=y` to maintain class balance  

3️⃣ **Model Training:**  
   - **Random Forest:** `RandomForestClassifier(n_estimators=100, random_state=42)`  
   - **XGBoost:** `XGBClassifier(use_label_encoder=False, eval_metric="logloss")`  

4️⃣ **Cross-Validation:**  
   - Using **5-Fold Stratified Cross-Validation**  

5️⃣ **Evaluation Metrics:**  
   - Accuracy, Precision, Recall, F1-score, Confusion Matrix  

---

## 📊 **Results & Performance**
| Model          | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|----------|--------|----------|
| Random Forest | 99.75%   | 92.5%    | 89.2%  | 90.8%    |
| XGBoost       | 99.82%   | 94.1%    | 91.3%  | 92.7%    |

✅ **XGBoost performed slightly better due to its ability to handle imbalanced data effectively.**  

---


---

## 🎯 **Key Takeaways**
- **Handling imbalanced data is crucial** for fraud detection.
- **Cross-validation helps** improve generalization.
- **XGBoost outperforms Random Forest** in this dataset.

---

## 🏆 **Future Improvements**
🔹 **Feature Engineering:** Adding more transaction-related features.  
🔹 **Deep Learning:** Trying LSTMs or Autoencoders.  
🔹 **Real-time Fraud Detection:** Deploying as a web API.  

---

## 🤝 **Contributing**
We welcome contributions! Feel free to **fork** this repository, make changes, and submit a **pull request**.

---

## 📝 **License**
This project is licensed under the **MIT License**.

---



# 📝 Important Questions & Answers on Credit Card Fraud Detection

## 1. What is credit card fraud detection, and why is it important?
Credit card fraud detection is the process of identifying unauthorized transactions on a credit card account. It is crucial to prevent financial losses for both banks and consumers. Fraudulent transactions can lead to huge monetary damages and erode customer trust in financial institutions.

## 2. What dataset is used for credit card fraud detection?
The dataset used in this project is the **Kaggle Credit Card Fraud Dataset**. It contains anonymized transaction data with labeled instances of fraud (1) and non-fraud (0). The dataset is highly imbalanced, with only about **0.17% fraudulent transactions**.

## 3. What challenges are involved in credit card fraud detection?
- **Imbalanced data:** The majority of transactions are legitimate, making fraud cases hard to detect.
- **Real-time detection:** Fraud detection needs to be fast to prevent further unauthorized transactions.
- **Feature engineering:** Fraudulent patterns may not be explicitly visible.
- **False positives:** Incorrectly flagging legitimate transactions can annoy users and damage trust.

## 4. How do you handle imbalanced datasets in fraud detection?
Several techniques are used:
- **Resampling methods**: Oversampling the minority class using SMOTE or undersampling the majority class.
- **Using different metrics**: Precision, recall, and F1-score instead of accuracy.
- **Cost-sensitive learning**: Assigning higher penalties to misclassifying fraud cases.

## 5. What machine learning models are used in this project?
The project uses **Random Forest** and **XGBoost** classifiers:
- **Random Forest**: An ensemble of decision trees that reduces overfitting.
- **XGBoost**: A gradient boosting method optimized for speed and performance.

## 6. How do you preprocess the dataset before training the models?
- **Feature scaling** using StandardScaler.
- **Handling missing values** if any.
- **Splitting data** into training and test sets.
- **Applying SMOTE** to balance classes if necessary.

## 7. How does Random Forest work in fraud detection?
Random Forest is an ensemble learning method that builds multiple decision trees and merges their outputs. It helps in reducing variance and improving accuracy. Since it handles imbalanced data well, it is useful for fraud detection.

## 8. Why is XGBoost used in fraud detection?
XGBoost is a boosting algorithm that builds trees sequentially, learning from previous mistakes. It is efficient, handles large datasets well, and is known for its superior performance in fraud detection problems.

## 9. What evaluation metrics are used in fraud detection?
Since fraud detection deals with imbalanced data, **accuracy alone is not a good metric**. Instead, we use:
- **Precision**: Measures how many detected frauds were actually fraud.
- **Recall**: Measures how many actual frauds were detected.
- **F1-score**: Harmonic mean of precision and recall.
- **Confusion Matrix**: Shows correct and incorrect predictions.

## 10. What were the results of Random Forest and XGBoost in this project?
| Model          | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|----------|--------|----------|
| Random Forest | 99.75%   | 92.5%    | 89.2%  | 90.8%    |
| XGBoost       | 99.82%   | 94.1%    | 91.3%  | 92.7%    |

XGBoost performed slightly better due to its ability to handle imbalanced data.

## 11. What is cross-validation, and why is it important?
Cross-validation ensures the model generalizes well to new data. In this project, **5-fold stratified cross-validation** is used to train the model on multiple data splits and evaluate its performance reliably.

## 12. What hyperparameter tuning was done for the models?
- **Random Forest**: Number of trees (`n_estimators`), depth (`max_depth`), and bootstrap sampling.
- **XGBoost**: Learning rate (`eta`), number of estimators (`n_estimators`), and tree depth (`max_depth`).

## 13. How can feature engineering improve fraud detection?
Feature engineering helps by creating more informative features such as:
- **Transaction velocity**: Frequency of transactions in a short period.
- **Location-based fraud**: Unusual locations for transactions.
- **Time-of-day analysis**: Fraudulent transactions often occur at odd hours.

## 14. What are the limitations of this project?
- **Static dataset**: Does not include real-time transactions.
- **Limited features**: Anonymized dataset means missing critical details.
- **Lack of behavioral analysis**: Cannot analyze customer transaction patterns over time.

## 15. How can deep learning be applied to fraud detection?
Deep learning methods like **Autoencoders and LSTMs** can detect anomalies in transaction sequences. These models can learn from vast amounts of historical data to predict fraudulent patterns.

## 16. How can this project be deployed in a real-world system?
- **Convert model into an API** using Flask or FastAPI.
- **Deploy on cloud platforms** like AWS, Azure, or Google Cloud.
- **Integrate with banking systems** to detect fraud in real-time.

## 17. How does anomaly detection differ from supervised learning in fraud detection?
- **Supervised learning**: Requires labeled fraud data to train models.
- **Anomaly detection**: Identifies unusual patterns without prior labels. Useful when fraud cases are rare.

## 18. What ethical considerations should be kept in mind?
- **False positives**: Flagging legitimate transactions can inconvenience users.
- **Bias in data**: Skewed datasets can lead to unfair decision-making.
- **Data privacy**: Sensitive credit card data must be handled securely.

## 19. What are some real-world fraud detection applications?
- **Banking transactions**: Identifying unauthorized purchases.
- **E-commerce fraud**: Detecting fake accounts or chargebacks.
- **Insurance fraud**: Preventing fake claims.
- **Government fraud detection**: Identifying tax frauds or benefit scams.

## 20. How can this project be extended?
- **Use deep learning models** like CNNs or RNNs for better pattern recognition.
- **Integrate real-time streaming** with Apache Kafka or Spark.
- **Add behavioral analytics** to detect unusual spending habits dynamically.

---
