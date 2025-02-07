import streamlit as st
import pandas as pd
import joblib
import base64
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model and preprocessor
model = joblib.load("models/final_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

# Function to set a local background image
def set_bg_image(image_path):
    """Sets background image in Streamlit app."""
    with open(image_path, "rb") as f:
        img_bytes = f.read()  # Read the image file as bytes
        img_base64 = base64.b64encode(img_bytes).decode()  # Encode image to base64
        st.markdown(
            f"""
            <style>
                .stApp {{
                    background-image: url("data:image/png;base64,{img_base64}");
                    background-size: cover;
                    background-position: center;
                }}
            </style>
            """, unsafe_allow_html=True
        )

# Set the local background image (replace with your image file path)
set_bg_image("images/background_image.jpg")  # Local path to your image

# Sidebar navigation with enhanced styling
st.sidebar.title("Navigation")
st.sidebar.markdown("""
    <style>
        /* Sidebar container styling */
        .stSidebar {
            background-color: #f7f9fc; /* Light background */
            border-radius: 12px; /* Rounded corners */
            padding: 15px;
        }
        /* Title styling */
        .sidebar-title {
            font-size: 24px;
            font-weight: bold;
            color: #00796b;
            margin-bottom: 20px;
        }
        /* Box for navigation items */
        .nav-box {
            background-color: #e0f7fa;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 2px 2px 15px rgba(0, 0, 0, 0.1);
        }
        /* Styling for navigation items */
        .stRadio > label {
            display: block;
            margin: 10px 0;
            padding: 10px;
            font-size: 18px; /* Larger font */
            font-weight: 500;
            color: #00574b; /* Text color */
            border: 1px solid #b2ebf2; /* Box border */
            border-radius: 6px; /* Rounded corners */
            transition: all 0.3s ease;
            text-align: center; /* Center text */
        }
        /* Hover and active states */
        .stRadio > label:hover {
            background-color: #00796b;
            color: white;
        }
        .stRadio > label[data-selected="true"] {
            background-color: #00796b;
            color: white;
            font-weight: bold;
        }
    </style>
    <div class="nav-box">
        <div class="stRadio">
            <!-- Placeholder for Streamlit radio -->
        </div>
    </div>
""", unsafe_allow_html=True)

# Create a radio button for navigation
page = st.sidebar.radio("Select a Page", ["Home", "Medical Charges Prediction", "About"])

# Display pages based on selection
if page == "Home":
    # Home page content
    st.title("Welcome to the Medical Insurance Charges Prediction App")
    st.write("""
        This is a simple app where you can predict the medical charges for a patient based on 
        their medical information. 
        Navigate through the pages using the sidebar to learn more or make a prediction.
    """)

elif page == "Medical Insurance Charges Prediction ":
    # Main page: Medical Charges Prediction form
    st.title("Medical Insurance Charges Prediction App")
    st.header("Input Patient Data")
    
    # User input form with validation
    age = st.number_input("Age", min_value=0, max_value=100, step=1, value=18)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1, value=20.0)
    children = st.number_input("Number of Children", min_value=0, max_value=10, step=1)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    if bmi <= 0:
        st.error("BMI must be a positive number!")
    elif age <= 0:
        st.error("Age must be a positive number!")
    else:
        if st.button("Predict"):
            # Create input data
            input_data = pd.DataFrame({
                "age": [age],
                "sex": [sex],
                "bmi": [bmi],
                "children": [children],
                "smoker": [smoker],
                "region": [region]
            })

            # Preprocess the input data
            input_data_preprocessed = preprocessor.transform(input_data)

            # Predict
            prediction = model.predict(input_data_preprocessed)

            # Display result
            st.success(f"Predicted Medical Insurance Charges : ${prediction[0]:.2f}")
            
            # Add a chart for interactivity (example)
            st.subheader("Medical Insurance Charges based on Age")
            charges_data = {
                'age': [30, 40, 50, 60, 70],
                'charges': [1200, 2000, 2500, 3000, 3500]  # Example charges
            }
            df = pd.DataFrame(charges_data)
            plt.figure(figsize=(8, 6))
            sns.barplot(x='age', y='charges', data=df)
            plt.title("Estimated Medical Charges based on Age")
            st.pyplot(plt)

elif page == "About":
    # About page content (displayed at the top of the page)
    st.title("About the App")
    
    # Creator Information at the top of the About section
    st.subheader("Created by:")
    st.write("### Rayhan Mahmud Ansari")
    st.write("Dept. of CSE, Sylhet Engineering College")
    st.write("Email: rayhan_mahmud@sec.ac.bd")
    
    st.write("### Nurul Islam Opu")
    st.write("Dept. of CSE, Sylhet Engineering College")
    st.write("Email: nurulislamopu1@gmail.com")
    
    # About the app content
    st.write("""
        This app uses machine learning to predict Medical Insurance Charges based on the patient's data.
        The model was trained on real insurance data collected from kaggle, and it takes into account factors like age, 
        BMI, smoking habits, and more to estimate the medical charges.
        
        **Features:**
        - Input a patient's information
        - Get a prediction of the medical charges
        - Learn about how the prediction is made on the About page.
    """)
    
    # Model Version
    st.write("### Model Version: 1.0.0")
