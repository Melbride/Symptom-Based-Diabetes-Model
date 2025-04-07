# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE, RandomOverSampler

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, roc_auc_score
from sklearn.feature_selection import SelectKBest, chi2
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

# Set page config
st.set_page_config(
    page_title="Diabetes Symptoms Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4169E1;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4682B4;
    }
    .section {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>Diabetes Detection App</h1>", unsafe_allow_html=True)
st.markdown("This app analyzes diabetes symptoms data and predicts the likelihood of diabetes based on input parameters.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('diabetes_symptoms_data.csv')
    cleaned_df = pd.read_csv('cleaned_symptoms_data.csv')
    return df, cleaned_df

# Function for EDA
def explore_data(df, cleaned_df):
    st.markdown("<h2 class='sub-header'>📊 Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    
    # Show first few rows
    if st.checkbox("Show sample data"):
        st.write(df.head())
    
    # Basic info
    if st.checkbox("Dataset information"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset shape:**", df.shape)
            st.write("**Missing values:**", df.isnull().sum().sum())
            
        with col2:
            # Class distribution
            class_counts = df['class'].value_counts()
            st.write("**Class distribution:**")
            st.write(f"Positive: {class_counts['Positive']} ({class_counts['Positive']/len(df):.1%})")
            st.write(f"Negative: {class_counts['Negative']} ({class_counts['Negative']/len(df):.1%})")
    
    # Gender distribution
    if st.checkbox("Gender distribution"):
        fig = px.pie(df, names='gender', title='Gender Distribution', 
                     color_discrete_sequence=px.colors.sequential.Blues)
        st.plotly_chart(fig)
    
    # Age distribution
    if st.checkbox("Age distribution"):
        fig = px.histogram(df, x="age", color="class", marginal="box", 
                           title="Age Distribution by Diabetes Status",
                           color_discrete_map={"Positive": "#1E90FF", "Negative": "#D3D3D3"})
        st.plotly_chart(fig)
    
    # Symptoms distribution
    if st.checkbox("Symptoms distribution"):
        # Create a melted dataframe for symptoms
        symptom_cols = ['polyuria', 'polydipsia', 'sudden_weight_loss', 'weakness', 
                         'polyphagia', 'genital_thrush', 'visual_blurring', 'itching', 
                         'irritability', 'delayed_healing', 'partial_paresis', 
                         'muscle_stiffness', 'alopecia', 'obesity']
        
        symptoms_by_class = pd.DataFrame()
        for symptom in symptom_cols:
            positive_yes = df[df['class'] == 'Positive'][symptom].value_counts().get('Yes', 0)
            positive_total = len(df[df['class'] == 'Positive'])
            positive_pct = positive_yes / positive_total * 100
            
            negative_yes = df[df['class'] == 'Negative'][symptom].value_counts().get('Yes', 0)
            negative_total = len(df[df['class'] == 'Negative'])
            negative_pct = negative_yes / negative_total * 100
            
            temp_df = pd.DataFrame({
                'Symptom': [symptom, symptom],
                'Class': ['Positive', 'Negative'],
                'Percentage': [positive_pct, negative_pct]
            })
            symptoms_by_class = pd.concat([symptoms_by_class, temp_df])
        
        fig = px.bar(symptoms_by_class, x='Symptom', y='Percentage', color='Class', barmode='group',
                     title='Percentage of "Yes" Responses by Symptom and Diabetes Status',
                     color_discrete_map={"Positive": "#1E90FF", "Negative": "#D3D3D3"})
        fig.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig)
    
    # Correlation heatmap
    if st.checkbox("Feature correlation"):
        # Convert Yes/No to 1/0 for correlation analysis
        df_encoded = df.copy()
        for col in df.select_dtypes(include=['object']).columns:
            if col != 'gender':  # Skip gender for now
                df_encoded[col] = df_encoded[col].map({'Yes': 1, 'No': 0})
        
        # One-hot encode gender
        df_encoded = pd.get_dummies(df_encoded, columns=['gender'], drop_first=True)
        
        # Map class to 1/0
        df_encoded['class'] = df_encoded['class'].map({'Positive': 1, 'Negative': 0})
        
        # Calculate correlation matrix
        corr = df_encoded.corr()
        
        # Plot heatmap
        fig = px.imshow(corr, text_auto=True, aspect="auto", 
                       title="Feature Correlation Heatmap",
                       color_continuous_scale='Blues')
        st.plotly_chart(fig)
    
    # Feature importance by chi-square test
    if st.checkbox("Feature importance"):
        # Prepare data for chi-square test
        X = cleaned_df.drop('class', axis=1)
        y = cleaned_df['class']
        
        # Select features with chi-square test
        selector = SelectKBest(chi2, k='all')
        selector.fit(X, y)
        feature_scores = pd.DataFrame({
            'Feature': X.columns,
            'Score': selector.scores_
        })
        
        # Sort by importance
        feature_scores = feature_scores.sort_values('Score', ascending=False)
        
        # Plot importance
        fig = px.bar(feature_scores, x='Feature', y='Score', 
                     title='Feature Importance (Chi-Square Test)',
                     color='Score', color_continuous_scale='Blues')
        fig.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig)

# Function for model building
def build_model(cleaned_df):
    st.markdown("<h2 class='sub-header'>🔍 Model Selection & Training</h2>", unsafe_allow_html=True)

    # Organize UI into two tabs
    tab1, tab2 = st.tabs(["🚀 Model Selection & Training", "📊 Evaluation"])

    with tab1:
        st.markdown("### Choose a Model and Train")
        model_option = st.radio("Select a Model:", ["🔵 Logistic Regression", "🌲 Random Forest"], index=None, horizontal=True)
        st.markdown("---")

        model = None  # Ensure model is always defined
        selected_features = ['age', 'gender', 'polyuria', 'polydipsia', 'sudden_weight_loss',
                             'polyphagia', 'irritability', 'partial_paresis']
        
        if model_option:
            # Store selected model in session state
            st.session_state['selected_model'] = model_option

            # Select relevant features
            X = cleaned_df[selected_features]
            y = cleaned_df["class"]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Apply SMOTE on training data only
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            # Scale age column (if it exists)
            if 'age' in X_train.columns:
                scaler = StandardScaler()
                X_train_resampled['age'] = scaler.fit_transform(X_train_resampled[['age']])
                X_test['age'] = scaler.transform(X_test[['age']])

            # Define the Model
            if model_option == "🔵 Logistic Regression":
                model = LogisticRegression(max_iter=1000, random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=30, max_depth=3, min_samples_split=15, min_samples_leaf=8, max_features="sqrt", class_weight="balanced", random_state=42)

            # Train the model
            model.fit(X_train_resampled, y_train_resampled)

            # Store trained model and data in session state
            st.session_state['model_trained'] = True
            st.session_state['trained_model'] = model
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['feature_names'] = X.columns  # Save feature names in session state

    with tab2:
        if 'model_trained' not in st.session_state or not st.session_state['model_trained']:
            st.warning("⚠️ Please select and train a model first!")
        else:
            st.markdown("### 📊 Evaluation Results")

            # Retrieve stored model and data
            model = st.session_state['trained_model']
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']

            # Evaluate Performance
            y_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.metric("🎯 Test Accuracy", f"{test_acc:.4f}")
                st.markdown("---")
                cv_scores = cross_val_score(model, X_test, y_test, cv=10)
                st.metric("📊 Cross-Validation Accuracy", f"{cv_scores.mean():.4f}")
                st.markdown("---")
                st.write("**Classification Report:**")
                st.dataframe(pd.DataFrame(report).transpose().style.format('{:.4f}'))

            with col2:
                st.markdown("**Confusion Matrix:**")
                fig = px.imshow(cm, text_auto=True, aspect="auto", color_continuous_scale="Blues")
                fig.update_layout(width=350, height=350)
                st.plotly_chart(fig)

            # Feature Importance for Random Forest
            if st.session_state.get('selected_model') == "🌲 Random Forest":
                st.markdown("### 🔍 Feature Importance")
                feature_importance = pd.Series(model.feature_importances_, index=selected_features).sort_values(ascending=False)
                fig_importance = px.bar(x=feature_importance.index, y=feature_importance.values,
                                        labels={'x': 'Feature', 'y': 'Importance'},
                                        title="Feature Importance in Random Forest Model")
                st.plotly_chart(fig_importance)

    return (st.session_state['trained_model'], st.session_state['feature_names']) if 'trained_model' in st.session_state else (None, None) 



# Function for prediction
def make_prediction(model, feature_names):
    st.markdown("<h2 class='sub-header'>🔮 Diabetes Detection</h2>", unsafe_allow_html=True)
    st.write("Enter the patient's information to get a prediction:")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Enter Age:", min_value=1, max_value=120, step=1, format="%d")
        gender = st.selectbox("Select Gender:", ["", "Male", "Female"])  # Placeholder for empty selection
        polyuria = st.selectbox("Do you experience Polyuria (Excessive Urination)?", ["", "Yes", "No"])
        polydipsia = st.selectbox("Do you experience Polydipsia (Excessive Thirst)?", ["", "Yes", "No"])
    
    with col2:
        sudden_weight_loss = st.selectbox("Have you had Sudden Weight Loss?", ["", "Yes", "No"])
        polyphagia = st.selectbox("Do you experience Polyphagia (Excessive Hunger)?", ["", "Yes", "No"])
        irritability = st.selectbox("Do you feel Irritability often?", ["", "Yes", "No"])
        partial_paresis = st.selectbox("Do you have Partial Paresis (Muscle Weakness)?", ["", "Yes", "No"])

    # Function to preprocess inputs
    def preprocess_inputs():
        if not age or not gender or not polyuria or not polydipsia or not sudden_weight_loss or not polyphagia or not irritability or not partial_paresis:
            st.warning("⚠️ Please fill in all fields before predicting.")
            return None

        gender_val = 1 if gender == "Male" else 0
        polyuria_val = 1 if polyuria == "Yes" else 0
        polydipsia_val = 1 if polydipsia == "Yes" else 0
        sudden_weight_loss_val = 1 if sudden_weight_loss == "Yes" else 0
        polyphagia_val = 1 if polyphagia == "Yes" else 0
        irritability_val = 1 if irritability == "Yes" else 0
        partial_paresis_val = 1 if partial_paresis == "Yes" else 0

        return np.array([[age / 120, gender_val, polyuria_val, polydipsia_val, sudden_weight_loss_val, 
                          polyphagia_val, irritability_val, partial_paresis_val]])

    # Prediction Button
    if st.button("🔍 Detect Diabetes Stage"):
        input_data = preprocess_inputs()

        if input_data is not None:
            with st.spinner("Calculating prediction..."):
                # Get Prediction Probability
                prediction_proba = model.predict_proba(input_data)
                diabetes_prob = prediction_proba[0][1]  # Probability of having diabetes
                
                # Determine risk category
                if diabetes_prob < 0.3:
                    risk_level = "Low Risk"
                    color = "#33FF57"
                    message = "✅ No Diabetes Detected!"
                elif 0.3 <= diabetes_prob < 0.5:
                    risk_level = "Moderate Risk"
                    color = "#FFD733"
                    message = "⚠️ Mild Diabetes (Borderline Case) - Monitor Your Health"
                elif 0.5 <= diabetes_prob < 0.8:
                    risk_level = "High Risk"
                    color = "#FF8C33"
                    message = "🚨 High Risk of Diabetes - Take Preventive Measures"
                else:
                    risk_level = "Very High Risk"
                    color = "#FF5733"
                    message = "🚨 Diabetes Detected - Consult a Doctor Immediately"

                # Display result
                st.markdown(f"<div style='background-color:#F0F8FF; padding:20px; border-radius:10px;'>", unsafe_allow_html=True)
                st.subheader("Prediction Result:")
                st.markdown(f"<h3 style='color:{color}'>{message}</h3>", unsafe_allow_html=True)
                st.write(f"📊 **Model Confidence:** {diabetes_prob*100:.2f}%")
                st.markdown(f"<h4 style='color:{color}'>Risk Level: {risk_level}</h4>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Recommendations
                st.subheader("Recommendations:")
                if diabetes_prob >= 0.5:
                    st.error("🚨 High Risk of Diabetes - Take Preventive Measures")
                    st.write("- **Consult a healthcare provider** for a diabetes assessment.")
                    st.write("- **Consider an HbA1c test** and **fasting blood glucose test**.")
                    st.write("- **Monitor blood sugar levels regularly**.")
                    st.write("- **Adopt a healthy diet and exercise routine**.")

                elif 0.3 <= diabetes_prob < 0.5:
                    st.warning("⚠️ Mild Diabetes (Borderline Case) - Monitor Your Health")
                    st.subheader("Recommendations for Mild Diabetes:")
                    st.write("- **Adopt a Healthy Diet:** Reduce sugar intake and eat fiber-rich foods.")
                    st.write("- **Exercise Regularly:** Engage in at least 30 minutes of moderate activity daily.")
                    st.write("- **Monitor Blood Sugar:** Keep track of glucose levels and consult a doctor if needed.")
                    st.write("- **Stay Hydrated:** Drink plenty of water to help with metabolism.")
                    st.write("- **Manage Stress:** Practice relaxation techniques like meditation or yoga.")

                else:
                    st.success("✅ No Diabetes Detected! Keep Up the Healthy Lifestyle")
                    st.write("- **Maintain a healthy lifestyle**.")
                    st.write("- **Regular check-ups with a healthcare provider**.")
                    st.write("- **Stay aware of diabetes symptoms and risk factors**.")

                # Disclaimer
                st.info("Disclaimer: This detection results are based on a machine learning model and should not be considered as medical advice. Always consult with a healthcare professional for proper diagnosis and treatment.")
# Information section
def show_info():
    st.markdown("<h2 class='sub-header'>ℹ️ About Diabetes</h2>", unsafe_allow_html=True)
    
    tabs = st.tabs(["What is Diabetes?", "Risk Factors", "Symptoms", "Prevention"])
    
    with tabs[0]:
        st.write("""
        **Diabetes** is a chronic disease that occurs either when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces. Insulin is a hormone that regulates blood sugar.
        
        There are two main types of diabetes:
        - **Type 1 Diabetes**: The body does not produce insulin. People with type 1 diabetes need daily insulin injections to control their blood glucose levels.
        - **Type 2 Diabetes**: The body does not use insulin effectively. This is the most common type of diabetes and is largely the result of excess body weight and physical inactivity.
        """)
    
    with tabs[1]:
        st.write("""
        **Risk factors for Type 2 Diabetes include:**
        - Family history of diabetes
        - Overweight or obesity
        - Physical inactivity
        - Age (risk increases with age)
        - High blood pressure
        - History of gestational diabetes
        - Polycystic ovary syndrome
        - History of heart disease or stroke
        - Certain ethnicities (including African American, Latino, Native American, and Asian American)
        """)
    
    with tabs[2]:
        st.write("""
        **Common symptoms of diabetes include:**
        - Polyuria (frequent urination)
        - Polydipsia (excessive thirst)
        - Polyphagia (excessive hunger)
        - Unexpected weight loss
        - Fatigue
        - Blurred vision
        - Slow-healing wounds
        - Frequent infections
        - Tingling or numbness in hands or feet
        
        Note that many people with Type 2 diabetes may not experience symptoms for years.
        """)
    
    with tabs[3]:
        st.write("""
        **Prevention strategies for Type 2 Diabetes:**
        - Maintain a healthy weight
        - Regular physical activity (at least 30 minutes per day)
        - Healthy diet rich in fruits, vegetables, and whole grains
        - Limit sugar and saturated fat intake
        - Don't smoke
        - Limit alcohol consumption
        - Regular health check-ups
        """)

# Main function
def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/diabetes.png", width=100)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Model Training", "Make Prediction", "About Diabetes"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Note:** This app is for educational purposes only and should not be used as a substitute for professional medical advice.
    """)
    
    # Load data
    df, cleaned_df = load_data()
    
    # Home page
    if page == "Home":
        st.image("https://img.icons8.com/color/96/000000/diabetes.png", width=150)
        st.markdown("<h1 class='main-header'>Welcome to the Diabetes Prediction App</h1>", unsafe_allow_html=True)
        
        st.write("""
        This application uses machine learning to detect the likelihood of diabetes based on various symptoms and risk factors.
        
        ### Features:
        - **Data Exploration**: Visualize and understand the diabetes symptoms dataset
        - **Model Training**: Train and evaluate machine learning models for diabetes prediction
        - **Make Prediction**: Input patient information to get a diabetes risk assessment
        - **About Diabetes**: Learn about diabetes, its symptoms, risk factors, and prevention strategies
        
        ### How to use:
        1. Navigate through the app using the sidebar
        2. Explore the data to understand diabetes risk factors
        3. Train a model to see how accurately we can predict diabetes
        4. Input patient information to get a prediction
        
        ### Dataset:
        The application uses a dataset containing information about various diabetes symptoms and their correlation with diabetes diagnosis.
        """)
        
        st.markdown("---")
        st.markdown("<h3>Quick Statistics</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Diabetes Positive", len(df[df['class'] == 'Positive']))
        with col3:
            st.metric("Diabetes Negative", len(df[df['class'] == 'Negative']))
    
    # Data Exploration page
    elif page == "Data Exploration":
        explore_data(df, cleaned_df)
    
    # Model Training page
    elif page == "Model Training":
        model, feature_names = build_model(cleaned_df)
        # Save the model and feature names in session state
        st.session_state['model'] = model
        st.session_state['feature_names'] = feature_names
    
    # Make Prediction page
    elif page == "Make Prediction":
        if 'model' not in st.session_state:
            st.warning("Please train a model first on the 'Model Training' page.")
            if st.button("Go to Model Training"):
                st.session_state['page'] = "Model Training"
                st.experimental_rerun()
        else:
            make_prediction(st.session_state['model'], st.session_state['feature_names'])
    
    # About Diabetes page
    # About Diabetes page
    elif page == "About Diabetes":
        show_info()
        
    # Add custom CSS
    st.markdown("""
    <style>
    .main-header {
        color: #2E86C1;
        font-size: 42px;
    }
    .sub-header {
        color: #3498DB;
    }
    </style>
    """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()
