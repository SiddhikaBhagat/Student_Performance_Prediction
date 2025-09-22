# Student Performance Prediction - End to End Machine Learning Project

##  Project Overview

End-to-end machine learning project that predicts student scores based on various demographic and academic factors. The project demonstrates a complete ML pipeline from data ingestion to model deployment using Flask web application.

##  Problem Statement

The goal is to understand how student performance (test scores) is affected by various factors like gender, ethnicity, parental level of education, lunch type, and test preparation course completion. We build a regression model to predict scores based on these features.

##  Project Structure

```
mlproject/
│
├── app.py                          # Flask web application
├── requirements.txt                
├── setup.py                       
├── README.md                      
│
├── artifacts/                     # Model artifacts and processed data
│   ├── data.csv                  
│   ├── train.csv                 
│   ├── test.csv                  
│   ├── model.pkl                 
│   └── preprocessor.pkl          
│
├── notebook/                     
│   ├── 1 . EDA STUDENT PERFORMANCE .ipynb
│   ├── 2. MODEL TRAINING.ipynb
│   └── data/
│       └── stud.csv             
│
├── src/                         # Source code package
│   ├── __init__.py
│   ├── exception.py             # Custom exception handling
│   ├── logger.py                # Logging configuration
│   ├── utils.py                 # Utility functions
│   │
│   ├── components/              # ML pipeline components
│   │   ├── __init__.py
│   │   ├── data_ingestion.py    # Data loading and splitting
│   │   ├── data_transformation.py # Feature engineering and preprocessing
│   │   └── model_trainer.py     # Model training and evaluation
│   │
│   └── pipeline/                # Prediction pipelines
│       ├── __init__.py
│       ├── predict_pipeline.py  # Inference pipeline
│       └── train_pipeline.py    # Training pipeline
│
└── templates/                   
    ├── home.html               
    └── index.html              
```

##  Project Workflow

### Step 1: Data Ingestion 
- Loads the student performance dataset.
- Splits data into training (80%) and testing (20%).
- Saves raw data, training data, and testing data to `artifacts/` folder
- Implements logging for tracking the ingestion process

### Step 2: Data Transformation
- **Numerical Features**: `reading_score`, `writing_score`
  - Applies median imputation for missing values
  - Standardizes features using StandardScaler
- **Categorical Features**: `gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`
  - Applies mode imputation for missing values
  - One-hot encoding for categorical variables
  - StandardScaler with `with_mean=False` for sparse matrices
- Creates a preprocessing pipeline using `ColumnTransformer`
- Saves the fitted preprocessor as `artifacts/preprocessor.pkl`

### Step 3: Model Training
- **Models Evaluated**:
  - Random Forest Regressor
  - Decision Tree Regressor
  - Gradient Boosting Regressor
  - Linear Regression
  - XGBoost Regressor
  - CatBoost Regressor
  - AdaBoost Regressor

- **Hyperparameter Tuning**: Uses `GridSearchCV` with 3-fold cross-validation
- **Model Selection**: Chooses the best model based on R² score on test data
- **Model Persistence**: Saves the best performing model as `artifacts/model.pkl`
- **Performance Threshold**: Ensures R² score > 0.6, otherwise raises exception

### Step 4: Prediction Pipeline
- **PredictPipeline Class**: 
  - Loads trained model and preprocessor
  - Applies preprocessing transformations
  - Makes predictions on new data
- **CustomData Class**: 
  - Handles input data formatting
  - Converts form inputs to DataFrame for prediction

### Step 5: Web Application
- **Flask Web Framework**: Creates user-friendly web interface
- **Routes**:
  - `/`: Landing page (`index.html`)
  - `/predictdata`: Prediction form and results (`home.html`)
- **Form Processing**: Handles user inputs for prediction
- **Real-time Prediction**: Displays math score predictions instantly

##  Technical Components

### Exception Handling
- Custom exception class for comprehensive error tracking
- Captures file names, line numbers, and error messages
- Provides detailed debugging information

### Logging
- Structured logging configuration
- Timestamped log files in `logs/` directory
- Tracks all major operations and errors

### Utility Functions
- **save_object()**: Serializes Python objects using pickle
- **load_object()**: Deserializes saved objects
- **evaluate_models()**: Compares multiple models with cross-validation

##  Dataset Information

**Source**: Student Performance Dataset
**Target Variable**: `math_score` (continuous)
**Features**:
- `gender`: Student's gender (male/female)
- `race_ethnicity`: Ethnic group (Group A-E)
- `parental_level_of_education`: Parent's education level
- `lunch`: Lunch type (standard/free or reduced)
- `test_preparation_course`: Test prep completion (none/completed)
- `reading_score`: Reading test score
- `writing_score`: Writing test score

**Dataset Size**: 1000+ student records


##  Key Learning Outcomes

This project demonstrates:
- End-to-end ML pipeline development
- Modular code architecture
- Exception handling and logging
- Model comparison and selection
- Object-oriented programming in ML
- Production-ready code practices

