# Car Price Prediction

## **Project Title**

**Car Price Prediction using Gradient Boosting Algorithm**

### **1. Problem Statement**

Predict the price of a car based on various features such as mileage, make, model, engine size, and year using a gradient boosting algorithm model.

### **2. Objective**

To develop a gradient boosting regression model that predicts car prices based on historical sales data and car features.

---

### **3. Data Processing and Extraction**

**Overview:** 

The dataset, initially in nested JSON format, has been transformed into a tabular format for ease of analysis.

[Extraction Details](https://www.notion.so/Extraction-2d3c5d173fab4c8a9e3eeb7bf9b2dc24?pvs=21)

**3.1 Dataset Overview**

- **Source:** Collected from various car listing datasets, including **carDekho**
- **Format:** CSV format with numerical and categorical features related to car specifications and sales.

**3.2 Features:**

- **Numerical Features:**
    - `Year`: Year of manufacture.
    - `km`: Total mileage.
    - `Ownerno`: Number of previous owners.
- **Categorical Features:**
    - `bt`: Body type (e.g., SUV, Hatchback).
    - `Model`: Specific model (e.g., Corolla, Mustang).
    - `FuelType`: Type of fuel (e.g., Gasoline, Diesel, Electric).

**3.3 Target Variable:**

- **Price:** The selling price of the car.

---

### **4. Data Preparation**

**4.1 Converting Lists and Dictionaries to DataFrames**

- Data was transformed from lists and dictionaries into DataFrames for better manipulation.

**4.2 Data Cleaning**

- **Handling Missing Values:**
    - Filled missing values for numerical features with median values and for categorical features with mode.

**4.3 Feature Engineering**

- **Categorical Encoding:**
    - Applied one-hot encoding to categorical features.
- **Data Type Conversion:**
    - Cleaned and converted data types for correct processing.

**4.4 Data Splitting**

- Split the dataset into 80% training and 20% testing.

---

### **5. Model Development**

**5.1 Model Selection**

- **Gradient Boosting Algorithm** was chosen for its capability to handle complex feature interactions.

**5.2 Model Training**

- The model was trained using `GradientBoostingRegressor` with specific hyperparameters.

**5.3 Hyperparameters:**

- n_estimators=300
- random_state=42
- learning_rate=0.2
- max_depth=5
- min_samples_leaf=4
- min_samples_split=2

---

### **6. Model Evaluation**

**6.1 Performance Metrics**

- **Mean Absolute Error (MAE):** 67,497.53
- **Mean Squared Error (MSE):** 10,206,216,625.31
- **R-squared (R²):** 0.92

**6.2 Key Features**

- `Body Type`, `Model Year`, `Transmission`, and `Mileage` are significant in predicting car prices.

---

### **7. Model Deployment**

**7.1 Deployment Method**

- Deployed as a **Streamlit** web app to allow users to input car features and receive price predictions.

---

### **8. Future Improvements**

- Explore hyperparameter tuning using GridSearchCV for further optimization.
- Evaluate additional models and techniques for improved performance.

---

### **9. References**

- Dataset Source: [CarDekho](https://www.cardekho.com/)
- Scikit-learn Documentation: [Scikit-learn](https://scikit-learn.org/)

---

### **Appendix**

- **Model Artifacts:** Trained model and preprocessor pipeline are saved as `.pkl` files.
- **Code Repository:** [GitHub Repository](https://github.com/parivallalShanmugam/car_dheko_price_pridiction.git)

### **Conclusion**

The gradient boosting model effectively predicts car prices, demonstrating robust performance with an R² value of 0.92. The model highlights the importance of features like `Body Type` and `Model Year`, which significantly influence pricing. Future work includes refining the model through hyperparameter tuning and exploring additional algorithms to enhance predictive accuracy. The deployment as a Streamlit app provides a user-friendly interface for real-time price estimation, making the model accessible to potential users and stakeholders.

