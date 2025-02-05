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

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", ["Home", "Medical Charges Prediction", "About"])

# Customizing the sidebar using markdown and CSS (only for sidebar)
st.sidebar.markdown("""
    <style>
        /* Sidebar Container */
        .stSidebar {
            background-color: #f7f9fc; /* Light background for a clean look */
            border-radius: 12px; /* Rounded corners */
            box-shadow: 2px 2px 15px rgba(0, 0, 0, 0.1); /* Subtle shadow for depth */
            padding: 15px;
        }

        /* Sidebar Title */
        .stSidebar .sidebar-title {
            font-size: 24px; /* Larger font for title */
            font-weight: bold;
            color: #00796b; /* Dark teal color */
            margin-bottom: 20px;
        }

        /* Navigation Links */
        .stRadio > label {
            display: block;
            margin: 8px 0; /* Space between items */
            padding: 10px;
            border-radius: 6px; /* Rounded corners for links */
            font-size: 18px; /* Larger font for better readability */
            font-weight: 500;
            color: #00574b; /* Slightly darker teal for link text */
            background-color: #e0f7fa; /* Light teal background */
            transition: all 0.3s ease; /* Smooth hover effect */
        }

        /* Hover and Active States */
        .stRadio > label:hover {
            background-color: #00796b; /* Dark teal on hover */
            color: white; /* White text on hover */
        }
        .stRadio > label[data-selected="true"] {
            background-color: #00796b; /* Dark teal for active */
            color: white; /* White text for active */
            font-weight: bold; /* Highlight active link */
        }
    </style>
""", unsafe_allow_html=True)

# Display pages based on selection
if page == "Home":
    # Home page content
    st.title("Welcome to the Medical Charges Prediction App")
    st.write("""
        This is a simple app where you can predict the medical charges for a patient based on 
        their medical information. 
        Navigate through the pages using the sidebar to learn more or make a prediction.
    """)

elif page == "Medical Charges Prediction":
    # Main page: Medical Charges Prediction form
    st.title("Medical Charges Prediction App")
    st.header("Input Patient Data")
    
    # User input form with validation
    age = st.number_input("Age", min_value=0, max_value=100, step=1, value=18)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1, )
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
            st.success(f"Predicted Medical Charges: ${prediction[0]:.2f}")
            
            # Add a chart for interactivity (example)
            st.subheader("Medical Charges based on Age")
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
        This app uses machine learning to predict medical charges based on the patient's data.
        The model was trained on real insurance data, and it takes into account factors like age, 
        BMI, smoking habits, and more to estimate the medical charges.
        
        **Features:**
        - Input a patient's information
        - Get a prediction of the medical charges
        - Learn about how the prediction is made on the About page.
    """)
    
    # Model Version
    st.write("### Model Version: 1.0.0")
