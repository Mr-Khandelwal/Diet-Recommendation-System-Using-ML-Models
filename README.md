# Diet-Recommendation-System-Using-ML-Models
A comprehensive project featuring a **calorie prediction system** using Gradient Boosting Classifier and ExtraTree Regressor for accurate predictions, alongside a **food recommendation system** leveraging TF-IDF and k-NN for personalized suggestions. Includes error analysis for performance insights.

# **Calorie Prediction and Food Recommendation System**

## **Overview**

This project combines two key functionalities:

1. **Calorie Prediction System**  
   Predicts the calorie content of food items using Gradient Boosting Classifier and ExtraTree Regressor for high accuracy and reliability.

2. **Food Recommendation System**  
   Recommends food items based on user preferences using TF-IDF vectorization and k-Nearest Neighbors (k-NN).

Both components aim to assist users in maintaining a balanced diet and making informed food choices.

---

## **Features**

### **Calorie Prediction**
- Predicts calorie content based on input features like food category, nutrients, etc.
- Implements Gradient Boosting Classifier for classification and ExtraTree Regressor for precise regression.

### **Food Recommendation**
- Utilizes TF-IDF vectorization to analyze food item descriptions.
- Employs k-NN to provide personalized food suggestions.

### **Error Analysis**
- Performance of models evaluated using metrics like MAE, MSE, RMSE, R² score.
- Includes detailed comparison and fine-tuning.

---

## **Installation**

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/calorie-food-system.git
   cd calorie-food-system
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the **Streamlit frontend**:
   ```bash
   streamlit run main.py
   ```

---

## **How It Works**

### **1. Calorie Prediction**
- Input food details via the Streamlit interface.
- The system predicts calorie content using trained Gradient Boosting and ExtraTree models.

### **2. Food Recommendation**
- Users can search for similar food items based on preferences.
- The system recommends items using cosine similarity and k-NN.

---

## **Dataset**
- Calorie data collected from reliable nutritional databases.
- Food item descriptions sourced from public datasets.

---

## **Technologies Used**
- **Python**: Core programming language.
- **Streamlit**: For building the frontend interface.
- **Machine Learning Models**: Gradient Boosting, ExtraTree, TF-IDF, and k-NN.
- **Pandas & NumPy**: Data preprocessing and manipulation.
- **Matplotlib & Seaborn**: Visualization of model performance.

---

## **Project Structure**

```
📂 calorie-food-system/
├── 📂 wweia_dataset/       # Contains datasets used for training and testing
├── 📂 models/              # Trained model files
├── 📂 notebooks/           # Jupyter notebooks for model training and analysis
├── 📂 src/                 # Core code files
├── main.py                 # Streamlit app for frontend
├── requirements.txt        # Required Python packages
└── README.md               # Project description
```

---

## **Performance Evaluation**
- Gradient Boosting Classifier: **XX% accuracy** (classification tasks).
- ExtraTree Regressor: MAE: **XX**, RMSE: **XX**.
- TF-IDF and k-NN: High recommendation relevance.

---

## **Future Work**
- Add support for more food categories.
- Integrate a real-time database for dynamic recommendations.

---

## **Contributors**
- **Rajat Khandelwal**  [LinkedIn](https://linkedin.com/in/your-profile)  

---

Feel free to contribute or raise issues! 😊
