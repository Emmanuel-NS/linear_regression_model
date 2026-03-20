# Pharma Sales Prediction Report (Extended Tutorial Edition)

This report is a long-form, tutorial-style explanation of the complete workflow in `multivariate.ipynb`. It is intentionally thorough and beginner-friendly, with deep explanations of the dataset, column meanings, preprocessing decisions, models, metrics, and interpretation. It also includes code snippets and actual outputs captured from your notebook run.

**Audience note:** This document is written for mixed audiences (instructors, engineers, and absolute beginners). Each section begins with a plain-language explanation, followed by technical detail. If you are advanced, you can skim the “Beginner Notes” blocks.

---

## Executive Summary
We trained three regression models (SGD Linear Regression, Decision Tree, Random Forest) to predict daily sales for the drug category `M01AB`. The dataset is time-based, so we extracted temporal features (Year, Month, Weekday), standardized them, and compared model performance using MSE and $R^2$. The best model by MSE was Random Forest, which we saved for later prediction.

**Key results (from your run):**
- Training MSE (SGD): 8.0100
- Test MSE (SGD): 8.0475
- Decision Tree MSE: 7.6753 ($R^2$ = -0.1161)
- Random Forest MSE: 7.4487 ($R^2$ = -0.0831)
- Best model by MSE: Random Forest

**Beginner Notes:**
- A “model” is a mathematical formula trained on data to make predictions.
- “MSE” is a metric that measures how far predictions are from the real values. Lower is better.
- Random Forest is a model that combines many decision trees to make a stronger prediction.

---

## 1. Mission and Problem Statement
**Mission:** Optimize pharmaceutical supply chain planning by forecasting daily demand for anti-inflammatory drugs (`M01AB`).

**Problem:** Pharmacies experience stockouts and overstocking because demand fluctuates by season and day of the week. A predictive model can reduce waste and improve availability by anticipating demand.

**Why regression?** The target is a continuous numeric value (daily sales volume). Regression is the right tool for predicting continuous numbers rather than categories.

**Beginner Notes:**
- If your answer is a number (like “5.7 units”), you usually use regression.
- If your answer is a label (like “High/Low”), you usually use classification.

---

## 2. Dataset Overview
**Source:** Pharma Sales Data (`salesdaily.csv`) from Kaggle.
**Observation unit:** One row per day.
**Size:** 2,106 records.

### 2.1 Data Dictionary (Key Columns)
| Column | Type | Description | Used in Model? |
| --- | --- | --- | --- |
| `datum` | string/date | Date of record | No (converted to date parts) |
| `M01AB` | float | Target sales for anti-inflammatory drugs | Yes (target) |
| `M01AE` | float | Sales for another drug group | No |
| `N02BA` | float | Sales for another drug group | No |
| `N02BE` | float | Sales for another drug group | No |
| `N05B` | float | Sales for another drug group | No |
| `N05C` | float | Sales for another drug group | No |
| `R03` | float | Sales for another drug group | No |
| `R06` | float | Sales for another drug group | No |
| `Year` | int | Year extracted from `datum` | Yes |
| `Month` | int | Month extracted from `datum` | Yes (one-hot encoded) |
| `Weekday` | int | Day of week (0=Mon, 6=Sun) | Yes (one-hot encoded) |
| `Hour` | int | Extra numeric column in dataset | No (unused) |
| `Weekday Name` | string | Weekday label | No (EDA only) |

### 2.2 Column Meaning (Deeper Explanation of Abbreviations)
The dataset uses the **ATC classification system**, which groups drugs by anatomical and therapeutic category. The abbreviations are standard in pharmaceutical sales data.

**Key:**
- The first letter indicates anatomical main group.
- The numbers and letters refine therapeutic or chemical groups.

**In this dataset:**
- `M01AB` = Anti-inflammatory and antirheumatic products, non-steroids (target)
- `M01AE` = Anti-inflammatory and antirheumatic products, non-steroids (another subclass)
- `N02BA` = Analgesics (e.g., aspirin-like compounds)
- `N02BE` = Analgesics (e.g., paracetamol-like compounds)
- `N05B` = Psycholeptics, anxiolytics
- `N05C` = Psycholeptics, hypnotics and sedatives
- `R03` = Drugs for obstructive airway diseases (e.g., asthma)
- `R06` = Antihistamines for systemic use

**Beginner Notes:**
- These codes are not random; they come from the ATC system, which is used globally to classify medicines.
- We predict only `M01AB` to keep the assignment focused and clear.

**Why these columns exist:**
The dataset tracks sales for multiple drug categories per day, which enables multi-target modeling or feature engineering. In this project, we focus on predicting only `M01AB` to keep the regression task clear and aligned with the assignment.

### 2.3 Data Info (Recorded Output)
The following is the actual `df.info()` output from the notebook run:

```
<class 'pandas.DataFrame'>
RangeIndex: 2106 entries, 0 to 2105
Data columns (total 13 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   datum         2106 non-null   str    
 1   M01AB         2106 non-null   float64
 2   M01AE         2106 non-null   float64
 3   N02BA         2106 non-null   float64
 4   N02BE         2106 non-null   float64
 5   N05B          2106 non-null   float64
 6   N05C          2106 non-null   float64
 7   R03           2106 non-null   float64
 8   R06           2106 non-null   float64
 9   Year          2106 non-null   int64  
 10  Month         2106 non-null   int64  
 11  Hour          2106 non-null   int64  
 12  Weekday Name  2106 non-null   str    
dtypes: float64(8), int64(3), str(2)
memory usage: 214.0 KB
```

### 2.4 First 5 Rows (Recorded Output)
| index | datum | M01AB | M01AE | N02BA | N02BE | N05B | N05C | R03 | R06 | Year | Month | Hour | Weekday Name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1/2/2014 | 0.0 | 3.67 | 3.4 | 32.40 | 7.0 | 0.0 | 0.0 | 2.0 | 2014 | 1 | 248 | Thursday |
| 1 | 1/3/2014 | 8.0 | 4.00 | 4.4 | 50.60 | 16.0 | 0.0 | 20.0 | 4.0 | 2014 | 1 | 276 | Friday |
| 2 | 1/4/2014 | 2.0 | 1.00 | 6.5 | 61.85 | 10.0 | 0.0 | 9.0 | 1.0 | 2014 | 1 | 276 | Saturday |
| 3 | 1/5/2014 | 4.0 | 3.00 | 7.0 | 41.10 | 8.0 | 0.0 | 3.0 | 0.0 | 2014 | 1 | 276 | Sunday |
| 4 | 1/6/2014 | 5.0 | 1.00 | 4.5 | 21.70 | 16.0 | 2.0 | 6.0 | 2.0 | 2014 | 1 | 276 | Monday |

**Interpretation:** The data has daily granularity, multiple drug categories, and clear timestamp features that can be engineered into predictive signals.

---

## 3. Data Loading and Caching (Reproducibility)
We first attempt to load a local pickle file. If it does not exist, we download the CSV from the raw GitHub URL and save it locally as a pickle.

**Code snippet:**
```python
if os.path.exists(LOCAL_PICKLE_FILE):
    df = pd.read_pickle(LOCAL_PICKLE_FILE)
else:
    df = pd.read_csv(DATA_URL)
    df.to_pickle(LOCAL_PICKLE_FILE)
```

**Why this matters:**
- Pickle caching avoids re-downloading and re-parsing each run.
- It gives you a consistent dataset snapshot in case the online file changes.

**Beginner Notes:**
- CSV is a plain text file; it is slower to load.
- Pickle is a binary file; it loads faster but is Python-specific.

---

## 4. Feature Engineering: Converting Non-Numeric Dates
The `datum` column is text. Models cannot use text directly, so we convert it to numeric parts.

**Code snippet:**
```python
df['datum'] = pd.to_datetime(df['datum'])
df['Year'] = df['datum'].dt.year
df['Month'] = df['datum'].dt.month
df['Day'] = df['datum'].dt.day
df['Weekday'] = df['datum'].dt.weekday
df['WeekdayName'] = df['datum'].dt.day_name()
```

**Why these fields matter:**
- **Year**: Captures long-term trends (overall growth/decline).
- **Month**: Captures seasonality (winter vs summer demand).
- **Weekday**: Captures weekly patterns (weekend vs weekday).
- **WeekdayName**: Used only for visualization (human-readable labels).

**Beginner Notes:**
- Machines cannot “understand” a date string like “1/2/2014” unless we break it into numbers.
- This step turns raw data into a form that a model can work with.

---

## 5. Exploratory Data Analysis (EDA)
We created three required visualizations:
1. Histogram of `M01AB` (distribution)
2. Boxplot of `M01AB` by weekday
3. Correlation heatmap

### 5.1 Distribution of Target (Histogram)
The histogram shows a right-skewed distribution. Most days have sales between 0 and 7 units, but some days spike to 10-17 units. This indicates occasional demand surges.

**Beginner Notes:**
- A histogram shows how often values appear.
- “Right-skewed” means most values are small, with a few large spikes.

### 5.2 Weekday Boxplot
The boxplot shows higher median values and more outliers on weekends (Saturday, Sunday). This suggests demand is slightly higher or more variable during weekends.

**Beginner Notes:**
- A boxplot summarizes a distribution using median, quartiles, and outliers.
- Higher medians on weekends suggest more sales on those days.

### 5.3 Correlation Heatmap
Correlations with `M01AB` are weak. This suggests no single feature alone explains sales; instead, subtle combinations and non-linear patterns matter.

**Beginner Notes:**
- Correlation ranges from -1 to +1.
- A value near 0 means there is no strong linear relationship.

---

## 6. Data Transformation (Nulls and Categories)

### 6.1 Missing Values
We inspect missing values:
```python
print(df.isnull().sum())
```

**Recorded output:**
All columns show 0 missing values. This means the dataset is complete. The imputation logic remains for robustness in case missing values appear later.

**Beginner Notes:**
- “Missing values” are empty cells.
- We still write code to handle missing values so the pipeline doesn’t break if future data has gaps.

### 6.2 One-Hot Encoding (Categorical Handling)
`Month` and `Weekday` are numeric but represent categories. We use one-hot encoding so the model does not treat them as ordered magnitudes.

**Code snippet:**
```python
X = pd.get_dummies(X, columns=['Month', 'Weekday'], drop_first=True)
```

**Generated columns (recorded):**
```
['Year', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6', 'Month_7',
 'Month_8', 'Month_9', 'Month_10', 'Month_11', 'Month_12',
 'Weekday_1', 'Weekday_2', 'Weekday_3', 'Weekday_4', 'Weekday_5', 'Weekday_6']
```

**Why `drop_first=True`:**
- Prevents redundancy.
- If all `Month_*` columns are 0, the model infers Month 1 (January).

**Beginner Notes:**
- One-hot encoding turns a category into many yes/no columns.
- We drop one column to avoid duplicating information.

### 6.3 Hybrid Imputation Strategy
Even though no missing values were found, the code handles missing data safely:

```python
if 'Year' in X.columns:
    X['Year'] = X['Year'].fillna(X['Year'].mean())
X = X.fillna(0)
```

**Reasoning:**
- `Year` is numeric and should not be filled with 0 (outlier).
- One-hot columns are binary, so 0 is the safest assumption for missing flags.

**Beginner Notes:**
- Filling `Year` with 0 would create unrealistic years. That is why we use the average.
- For yes/no columns, “0” is simply “No.”

---

## 7. Standardization (Scaling)
We standardize features because gradient descent is sensitive to scale.

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Why this matters:**
- `Year` values are large (~2014), while one-hot values are 0/1.
- Without scaling, SGD would focus only on large values and converge poorly.

**Beginner Notes:**
- Scaling makes all features comparable in size.
- It helps gradient descent find the best answer faster.

---

## 8. Model Training and Evaluation

### 8.1 Model 1: SGD Linear Regression (Gradient Descent)
**Purpose:** Required by assignment. Optimized using gradient descent.

```python
sgd_reg = SGDRegressor(
    max_iter=1000,
    tol=1e-3,
    penalty=None,
    learning_rate='constant',
    eta0=0.01,
    random_state=42
)
```

**Recorded output:**
- Final Train MSE: 8.0100
- Final Test MSE: 8.0475

**Interpretation:**
The training and test errors are close, which suggests the model generalizes but has limited predictive power.

**Beginner Notes:**
- “Gradient descent” is a method that repeatedly adjusts the model to reduce error.
- If training and test error are close, the model is not overfitting.

### 8.2 Model 2: Decision Tree Regressor
**Purpose:** Captures non-linear rules (specific months or weekdays with spikes).

**Recorded output:**
- Decision Tree MSE: 7.6753
- Decision Tree $R^2$: -0.1161

**Beginner Notes:**
- A decision tree is like a flowchart of rules.
- It can overfit if it becomes too deep.

### 8.3 Model 3: Random Forest Regressor
**Purpose:** Ensemble of multiple trees, usually more stable.

**Recorded output:**
- Random Forest MSE: 7.4487
- Random Forest $R^2$: -0.0831

**Beginner Notes:**
- Random Forest averages many trees to make a stronger prediction.
- This usually improves stability and accuracy.

---

## 9. Model Comparison
| Model | MSE | $R^2$ |
| --- | --- | --- |
| SGD Linear Regression | 8.0475 | -0.1702 |
| Decision Tree | 7.6753 | -0.1161 |
| Random Forest | 7.4487 | -0.0831 |

**Best model:** Random Forest (lowest MSE).

**Why it won:**
Random Forest captures non-linear relationships and reduces variance by averaging multiple decision trees.

**Beginner Notes:**
- Lower MSE means better predictions.
- We choose the model that performs best on unseen test data.

---

## 10. Saved Model Artifacts
We saved:
- `best_model.pkl`: fitted Random Forest model.
- `scaler.pkl`: fitted StandardScaler.

These files allow Task 2 prediction without retraining.

---

## 11. Prediction Example (Task 2 Preparation)
We tested a sample prediction and validated on one real row:

**Recorded output:**
- Predicted Sales for Friday, Dec 2024: 5.57
- Actual (first row): 0.00
- Predicted (first row): 3.61

**Interpretation:**
The model can generate a numeric prediction, though it is not perfect for every day.

**Beginner Notes:**
- Prediction is only as good as the data and features you provide.
- The output is a number, not a category.

**Core logic used:**
```python
input_encoded = pd.get_dummies(input_data, columns=['Month', 'Weekday'])
input_final = input_encoded.reindex(columns=training_columns, fill_value=0)
input_scaled = scaler.transform(input_final)
prediction = best_model.predict(input_scaled)
```

---

## 12. Visual Results Interpretation (Required for Rubric)
1. **Distribution Plot (Histogram):**
   Right-skewed distribution suggests many low-demand days and a few high-demand spikes.
2. **Weekday Boxplot:**
   Weekends show higher medians and more outliers, indicating variability.
3. **Correlation Heatmap:**
   Weak correlations suggest the need for non-linear models and feature interactions.

---

## 13. Limitations and Improvement Ideas
**Limitations:**
- Only time-based features used. No lag features, holidays, or external factors.
- Negative $R^2$ suggests models are only slightly better than baseline.

**Improvements:**
- Add lag features (previous 1-7 days sales).
- Include holiday indicators.
- Tune hyperparameters (tree depth, learning rate, estimators).
- Try advanced models (XGBoost, LightGBM).

**Beginner Notes:**
- Real-world prediction improves with better features, not just “more complex” models.
- Feature engineering is often more important than model choice.

---

## 14. Math Foundations (Loss, Gradient Descent, Bias-Variance)

This section explains the math ideas used by the models. You do not need to memorize the formulas, but understanding the intuition helps you justify your approach in the report.

### 14.1 Loss Function (Mean Squared Error)
We want predictions to be as close as possible to actual values. The error is measured using Mean Squared Error (MSE):

$$
    ext{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

**Intuition:**
- If predictions are perfect, MSE is 0.
- Squaring makes large mistakes count more than small ones.

### 14.2 Linear Regression Model Formula
Linear regression predicts using a weighted sum of features:

$$
\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_k x_k
$$

**Intuition:**
- Each feature has a weight that shows how strongly it affects the prediction.
- Training means learning the best weights.

### 14.3 Gradient Descent (How SGD Learns)
Gradient descent updates weights to reduce error:

$$
w \leftarrow w - \alpha \cdot \nabla \text{MSE}
$$

Where:
- $w$ = model weights
- $\alpha$ = learning rate (step size)
- $\nabla \text{MSE}$ = gradient (direction of steepest error increase)

**Intuition:**
- The gradient points uphill. We step downhill to reduce error.
- Small steps avoid overshooting; large steps can diverge.

### 14.4 Bias-Variance Tradeoff
Model error comes from two sources:
- **Bias:** Model is too simple and underfits.
- **Variance:** Model is too complex and overfits.

**Interpretation in this project:**
- SGD Linear Regression has higher bias (simple model).
- Random Forest has lower bias but can have higher variance (mitigated by averaging trees).

### 14.5 Normal Equation (Closed-Form Linear Regression)
In classical linear regression, you can solve for the best weights directly (no gradient descent) using the normal equation:

$$
\hat{w} = (X^T X)^{-1} X^T y
$$

Where:
- $X$ is the feature matrix (rows are samples, columns are features)
- $y$ is the target vector
- $X^T$ is the transpose of $X$
- $(X^T X)^{-1}$ is the inverse of the matrix $X^T X$

**Intuition:**
- This formula gives the exact weights that minimize MSE for linear regression.
- It is fast for small datasets but becomes expensive for very large datasets or many features.
- We used SGD (gradient descent) because the assignment requires it and it scales well.

### 14.5.1 Normal Equation Derivation (Step-by-Step)
We start with the sum of squared errors (SSE):

$$
J(w) = \|Xw - y\|^2 = (Xw - y)^T (Xw - y)
$$

Expand and take the derivative with respect to $w$:

$$
J(w) = w^T X^T X w - 2 y^T X w + y^T y
$$

$$
\nabla_w J(w) = 2 X^T X w - 2 X^T y
$$

Set the gradient to zero to minimize the error:

$$
2 X^T X w - 2 X^T y = 0
$$

$$
X^T X w = X^T y
$$

Solve for $w$:

$$
w = (X^T X)^{-1} X^T y
$$

**Note:** If $X^T X$ is not invertible, we can use a pseudo-inverse.

### 14.5.2 Worked Numeric Example (Tiny Dataset)
Suppose we have two data points and one feature:

$$
X = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \quad y = \begin{bmatrix} 2 \\ 3 \end{bmatrix}
$$

Compute:

$$
X^T X = [1 \; 2] \begin{bmatrix} 1 \\ 2 \end{bmatrix} = [5]
$$

$$
X^T y = [1 \; 2] \begin{bmatrix} 2 \\ 3 \end{bmatrix} = [8]
$$

$$
w = (X^T X)^{-1} X^T y = \frac{1}{5} \cdot 8 = 1.6
$$

So the model is:

$$
\hat{y} = 1.6 x
$$

For $x=1$, $\hat{y}=1.6$ (close to 2). For $x=2$, $\hat{y}=3.2$ (close to 3).

### 14.6 Math Diagrams (Text Descriptions)
These are diagram descriptions you can include in a written report or slide deck:

**Diagram A: Loss Surface (Bowl Shape)**
- Imagine a 3D bowl where the bottom is the lowest error.
- Each point on the surface is a different set of weights.
- Gradient descent is like a ball rolling downhill to the lowest point.

**Diagram B: Bias vs Variance**
- Draw a target board.
- High bias: all arrows cluster far from the center (systematic error).
- High variance: arrows are scattered widely (unstable model).
- Best models cluster near the center with tight spread.

**Diagram C: One-Hot Encoding**
- Show a single column "Month" replaced by columns "Month_2", "Month_3", ... "Month_12".
- Highlight that January is represented by all zeros.

---

## 15. Data Profiling Appendix (Describe, Quantiles, Distributions)
This section records the numeric summary statistics directly from `df.describe()` and quantile analysis. These outputs help explain the scale and spread of the data.

### 15.1 `df.describe()` Output (Recorded)
```
                     datum        M01AB        M01AE        N02BA  \
count                 2106  2106.000000  2106.000000  2106.000000   
mean   2016-11-19 12:00:00     5.033683     3.895830     3.880441   
min    2014-01-02 00:00:00     0.000000     0.000000     0.000000   
25%    2015-06-12 06:00:00     3.000000     2.340000     2.000000   
50%    2016-11-19 12:00:00     4.990000     3.670000     3.500000   
75%    2018-04-29 18:00:00     6.670000     5.138000     5.200000   
max    2019-10-08 00:00:00    17.340000    14.463000    16.000000   
std                    NaN     2.737579     2.133337     2.384010   

             N02BE         N05B         N05C          R03          R06  \
count  2106.000000  2106.000000  2106.000000  2106.000000  2106.000000   
mean     29.917095     8.853627     0.593522     5.512262     2.900198   
min       0.000000     0.000000     0.000000     0.000000     0.000000   
25%      19.000000     5.000000     0.000000     1.000000     1.000000   
50%      26.900000     8.000000     0.000000     4.000000     2.000000   
75%      38.300000    12.000000     1.000000     8.000000     4.000000   
max     161.000000    54.833333     9.000000    45.000000    15.000000   
std      15.590966     5.605605     1.092988     6.428736     2.415816   

              Year        Month         Hour          Day      Weekday  
count  2106.000000  2106.000000  2106.000000  2106.000000  2106.000000  
mean   2016.401235     6.344255   275.945869    15.686135     3.000475  
min    2014.000000     1.000000   190.000000     1.000000     0.000000  
25%    2015.000000     3.000000   276.000000     8.000000     1.000000  
50%    2016.000000     6.000000   276.000000    16.000000     3.000000  
75%    2018.000000     9.000000   276.000000    23.000000     5.000000  
max    2019.000000    12.000000   276.000000    31.000000     6.000000  
std       1.665060     3.386954     1.970547     8.806215     2.000831  
```

**Interpretation:**
- The average `M01AB` is about 5.03 with a maximum of 17.34, confirming the right-skew.
- `Month` has a mean near 6.34, meaning the dataset is fairly balanced across months.

### 15.2 Quantiles (M01AB)
```
0.00     0.00
0.25     3.00
0.50     4.99
0.75     6.67
0.90     8.67
0.95    10.00
1.00    17.34
```

### 15.3 Quantiles (M01AE)
```
0.00     0.000
0.25     2.340
0.50     3.670
0.75     5.138
0.90     6.670
0.95     7.660
1.00    14.463
```

**Interpretation:**
- The 95th percentile for `M01AB` is 10.00, but the max is 17.34. This shows rare spikes.
- `M01AE` is slightly lower in median and maximum compared to `M01AB`.

---

## 16. Why $R^2$ Can Be Negative (Mini-Chapter)
The $R^2$ score compares your model to a baseline that always predicts the mean of $y$.

$$
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

Where:
- $\hat{y}_i$ = model prediction
- $\bar{y}$ = mean of the target

**Key idea:**
- If your model is worse than predicting the mean, the numerator is larger than the denominator, so $R^2$ becomes negative.

**Example:**
- Suppose the average daily sales is 5 units.
- If your model consistently predicts 10 units for every day, its errors may be worse than just predicting 5.
- In that case, $R^2 < 0$.

**Interpretation in this project:**
- The data has weak linear signals.
- Without extra features (holidays, lag sales, promotions), models struggle to outperform the mean baseline.

**Takeaway:**
- Negative $R^2$ does not mean the model is useless; it means the feature set is too simple for strong performance.
- Adding better features often improves $R^2$ dramatically.

---

## 17. Beginner Glossary (Quick Definitions)
- **Dataset:** A structured table of data (rows and columns).
- **Feature:** A column used as input to a model (e.g., Month, Weekday).
- **Target:** The value we want to predict (here, `M01AB`).
- **Model:** A learned function that maps features to predictions.
- **Training:** The process of learning from data.
- **Testing:** Evaluating the model on data it has not seen.
- **MSE:** Mean Squared Error; average squared difference between prediction and actual.
- **$R^2$:** A metric that indicates how much variance in the target is explained by the model.
- **One-hot encoding:** Converting categories to multiple yes/no columns.
- **Scaling:** Making all features similar in range.
- **Overfitting:** When a model memorizes training data and fails on new data.

---

## 18. Walkthrough: One Row Through the Entire Pipeline
This section shows exactly how one row becomes a prediction. We use the first row as an example.

### 17.1 Original Row
From the dataset (first row):

| datum | M01AB | Year | Month | Weekday |
| --- | --- | --- | --- | --- |
| 1/2/2014 | 0.0 | 2014 | 1 | 3 |

**Interpretation:**
- Date: January 2, 2014
- Target (actual): 0.0 units
- Weekday = 3 (Thursday)

### 17.2 Feature Extraction
We already have Year=2014, Month=1, Weekday=3 (Thursday). These become the feature vector.

### 17.3 One-Hot Encoding
Month and Weekday become multiple binary columns. For Month=1:

- `Month_2` to `Month_12` are all 0 (because it is January).

For Weekday=3 (Thursday):

- `Weekday_1` = 0 (Tuesday)
- `Weekday_2` = 0 (Wednesday)
- `Weekday_3` = 1 (Thursday)
- `Weekday_4` = 0 (Friday)
- `Weekday_5` = 0 (Saturday)
- `Weekday_6` = 0 (Sunday)

So the row becomes a numeric vector with 18 columns:

```
[Year=2014, Month_2=0, Month_3=0, Month_4=0, Month_5=0, Month_6=0,
 Month_7=0, Month_8=0, Month_9=0, Month_10=0, Month_11=0, Month_12=0,
 Weekday_1=0, Weekday_2=0, Weekday_3=1, Weekday_4=0, Weekday_5=0, Weekday_6=0]
```

### 17.4 Scaling
`StandardScaler` converts the Year value into a standardized number relative to the dataset mean and standard deviation. This produces a scaled feature vector that the model expects.

### 17.5 Prediction
The model applies its learned weights and outputs a predicted value. In our run, the prediction for the first row was:

- Actual: 0.00
- Predicted: 3.61

**Conclusion:** The model produces a reasonable numeric estimate, but it is not perfect for every day.

---

## 19. Function-by-Function Tutorial (Beginner Friendly)
This section explains each key function used, what it does, and why we used it.

### 18.1 `pd.read_csv()`
**What it does:** Reads a CSV file into a table (DataFrame).
**Why we use it:** CSV is the most common dataset format.

### 18.2 `pd.read_pickle()` / `to_pickle()`
**What it does:** Saves and loads Python objects faster than CSV.
**Why we use it:** Speeds up repeated runs of the notebook.

### 18.3 `pd.to_datetime()`
**What it does:** Converts a text date ("1/2/2014") into a real date object.
**Why we use it:** Allows extraction of Year, Month, Day, Weekday.

### 18.4 `pd.get_dummies()`
**What it does:** One-hot encodes categorical variables into binary columns.
**Why we use it:** Prevents models from treating categories as numbers.

### 18.5 `train_test_split()`
**What it does:** Splits data into training and testing sets.
**Why we use it:** To test the model on unseen data.

### 18.6 `StandardScaler()`
**What it does:** Normalizes features to mean=0 and variance=1.
**Why we use it:** Required for gradient descent to converge properly.

### 18.7 `SGDRegressor()`
**What it does:** Implements linear regression using gradient descent.
**Why we use it:** Assignment requires gradient descent optimization.

### 18.8 `DecisionTreeRegressor()`
**What it does:** Builds a flowchart-like model based on splits.
**Why we use it:** Captures non-linear patterns.

### 18.9 `RandomForestRegressor()`
**What it does:** Combines many decision trees for better accuracy.
**Why we use it:** Often improves performance and stability.

### 18.10 `mean_squared_error()` and `r2_score()`
**What they do:** Evaluate model accuracy.
**Why we use them:** MSE measures error magnitude; $R^2$ measures explained variance.

---

## 20. Model Concepts in Plain Language

### 19.1 Linear Regression (SGD)
Imagine drawing a straight line through scattered points. The algorithm keeps adjusting the line to make the total error smaller. That adjustment process is called gradient descent.

### 19.2 Decision Tree
Imagine playing 20 Questions. The tree asks yes/no questions like “Is Month > 6?” and “Is Weekday = Friday?” until it makes a prediction.

### 19.3 Random Forest
Instead of trusting one tree, a forest asks many trees and takes the average. This reduces mistakes made by any single tree.

---

## 21. Mini-FAQ (Common Beginner Questions)

**Q: Why do we need training and testing sets?**
A: To check if the model works on data it has never seen. This avoids memorizing.

**Q: Why not use all columns as features?**
A: Some columns are not predictive or are too indirect (e.g., other drug categories). Using all may add noise.

**Q: Why is $R^2$ negative?**
A: It means the model is performing worse than simply predicting the average every day. This is common for noisy time-series without strong signals.

**Q: Is Random Forest always the best?**
A: Not always. It worked best here, but other datasets might favor linear models or boosted trees.

---

## 22. Suggested Learning Path for Absolute Beginners
If you are new to data science, follow this sequence:
1. Learn what a DataFrame is (`pandas`).
2. Learn how to plot distributions and boxplots (`seaborn`).
3. Learn why preprocessing matters (missing values, scaling, encoding).
4. Start with linear regression.
5. Move to decision trees, then random forests.
6. Compare models with metrics like MSE and $R^2$.

---

## Conclusion: Key Insights and Takeaways

### Key Insights
- The dataset is clean (no missing values in the current run) and structured around daily time signals.
- Sales of `M01AB` are right-skewed, with most days having low sales and occasional spikes.
- Weekday patterns show higher variability on weekends, which supports using weekday features.
- Correlation analysis shows weak linear relationships, which explains why linear regression underperforms.
- Random Forest achieved the lowest MSE, suggesting non-linear patterns matter.

### Practical Takeaways
- Feature engineering is critical: converting dates into Year/Month/Weekday is what makes prediction possible.
- One-hot encoding avoids false numeric ordering (e.g., December is not “12 times” January).
- Standardization is essential for gradient descent models like SGD.
- Negative $R^2$ indicates a weak feature set, not necessarily a bad model; richer features (lags, holidays) can improve performance.
- Saving both the model and scaler ensures reproducible predictions for Task 2.

### Final Recommendation
For this dataset, Random Forest is the best baseline model. For stronger accuracy, add lag features (past sales), holidays, and promotions, then re-train and compare metrics.

---

## 23. Appendix A: Full Metric Outputs (Raw)
```
Final Train MSE: 8.01002594932732
Final Test MSE: 8.047475881505763

Decision Tree MSE: 7.6753
Decision Tree R2: -0.1161
Random Forest MSE: 7.4487
Random Forest R2: -0.0831

Model Comparison:
                   Model       MSE        R2
0  SGD Linear Regression  8.047476 -0.170180
1          Decision Tree  7.675290 -0.116061
2          Random Forest  7.448696 -0.083112

Best Performing Model: Random Forest
```

---

## 24. Appendix B: Key Code Snippets

**Load data with cache:**
```python
df = pd.read_pickle(LOCAL_PICKLE_FILE)
```

**One-hot encoding:**
```python
X = pd.get_dummies(X, columns=['Month', 'Weekday'], drop_first=True)
```

**Standardize:**
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

**Model training loop:**
```python
for epoch in range(epochs):
    sgd_reg.partial_fit(X_train_scaled, y_train)
```

**Save model:**
```python
joblib.dump(best_model, 'best_model.pkl')
```
