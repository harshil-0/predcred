# 🏦 Wallet Credit Scoring System

This project implements a **hybrid credit scoring model** for DeFi users based on their Aave V2 transaction history. It combines **rule-based heuristics** with a **machine learning model** to generate a credit score (0–1000) for each user wallet.

---

This will:

- ✅ Parse the JSON file into a DataFrame  
- 🧠 Extract wallet-level features  
- 📏 Apply rule-based scoring  
- 🤖 Train a machine learning model  
- 💾 Output scores to `wallet_scores.csv`  
- 📊 Print model evaluation and sample predictions  


## 📂 Files

- `user-wallet-transactions.json` – JSON input file with user-level DeFi activity.
- `wallet_scores.csv` – Output CSV containing credit scores per wallet.

---

## ✅ Credit Score Logic

### 🔎 Rule-Based Scoring

Each wallet starts with a **base score of 500**. The following features are calculated from Aave actions:

| Behavior                | Feature                            | Effect on Score            | Max Impact |
|-------------------------|-------------------------------------|-----------------------------|------------|
| 📥 Total Deposits       | `total_deposit_usd`                | +1 per $100 deposited       | +200       |
| 💸 Repayment Behavior   | `repay_to_borrow_ratio`            | +100 per full repayment     | +200       |
| 🔄 Redemptions          | `redeem_to_deposit_ratio`          | +100 per full redemption    | +100       |
| ⚠️ Liquidations         | `liquidation_ratio`                | −500 per full liquidation   | −200       |
| ❌ Unpaid Borrow        | `borrow - repay`                   | −1 per $100 unpaid          | −100       |

Final score is **clamped between 0 and 1000**.

---

### 🧠 ML-Based Scoring 
The model uses a `RandomForestRegressor` trained on user behavior features:

```python
[
    total_deposit_usd, total_borrow_usd, total_repay_usd,
    total_redeem_usd, total_liquidation_usd,
    num_deposits, num_borrows, num_repays,
    num_redeems, num_liquidations,
    repay_to_borrow_ratio, redeem_to_deposit_ratio, liquidation_ratio
]
```
## 🔍 Model Training and Testing Explained

The `train_credit_score_model` function handles the machine learning portion of the pipeline. Here's a breakdown of how the model is trained and evaluated:

### 🧱 Step 1: Data Preparation

- The `user_features_dict` (wallet-level features extracted from the transaction logs) is converted into a Pandas DataFrame.
- A new column `credit_score` is added by applying the rule-based scoring logic to generate **pseudo-labels**. These act as ground truth for supervised learning.
- The features (`X`) are selected by dropping `userWallet` and `credit_score`.
- The labels (`y`) are the generated credit scores.

### 🧪 Step 2: Train/Test Split

- The dataset is split into training and test sets using an 80/20 ratio.
- `X_train`, `y_train`: used for model training.
- `X_test`, `y_test`: held out for model evaluation.
- `wallet_train`, `wallet_test`: corresponding wallet IDs for analysis.

### 🌲 Step 3: Model Training

- A **Random Forest Regressor** (`RandomForestRegressor`) is used.
- It's trained on the training set (`X_train`, `y_train`) to learn how features map to credit scores.
- The model learns complex non-linear relationships between features such as deposit/borrow behavior and the final credit score.

### 📈 Step 4: Prediction and Evaluation

- The trained model is used to predict credit scores on the unseen test set:  
  `y_pred = model.predict(X_test)`
- The predicted scores are clamped between 0 and 1000 for realism.
- Evaluation metrics:
  - **MAE (Mean Absolute Error)**: Measures the average absolute error between predicted and actual scores.
  - **R² (R-squared)**: Indicates how well the model explains the variance in the data.
- A sample of the predictions is printed for quick inspection.

### 💡 Summary

| Component       | Description                                      |
|----------------|--------------------------------------------------|
| Model           | `RandomForestRegressor`                         |
| Target          | Rule-based `credit_score`                       |
| Input Features  | Wallet behavior: deposits, borrows, repayments  |
| Output          | `wallet_scores.csv` with predicted scores       |
| Metrics         | MAE, R² score                                   |

This hybrid approach allows combining expert rule logic with ML flexibility for future generalization.

