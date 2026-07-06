# Instagram Content Performance Prediction using Machine Learning

# Step 1: Import Libraries

import streamlit as st
import pandas as pd
import joblib

# Step 2: Load Saved Model & Encoders

model = joblib.load("instagram_model.pkl")
encoders = joblib.load("label_encoders.pkl")

# Step 3: Page Configuration

st.set_page_config(
    page_title="Instagram Content Performance Predictor",
    page_icon="📱",
    layout="centered"
)

# Step 4: Sidebar

st.sidebar.title("📘 Project Information")

st.sidebar.write("""
### AIML Mini Project

**Project Name**
Instagram Content Performance Prediction

**Algorithm**
Random Forest Classifier

**Model Accuracy**
85.13%

**Developed By**
Hrucha Patil
""")

# Step 5: Main Heading

st.title("📱 Instagram Content Performance Predictor")

st.write(
    "Predict whether an Instagram post will perform "
    "**Low**, **Medium**, **High** or **Viral** using Machine Learning."
)

st.markdown("---")

st.subheader("📝 Enter Instagram Post Details")

# Step 6: User Inputs

account_type = st.selectbox(
    "Account Type",
    encoders["account_type"].classes_
)

media_type = st.selectbox(
    "Media Type",
    encoders["media_type"].classes_
)

content_category = st.selectbox(
    "Content Category",
    encoders["content_category"].classes_
)

traffic_source = st.selectbox(
    "Traffic Source",
    encoders["traffic_source"].classes_
)

has_call_to_action = st.selectbox(
    "Call To Action",
    [0, 1]
)

post_hour = st.slider(
    "Post Hour",
    0,
    23,
    12
)

day_of_week = st.selectbox(
    "Day of Week",
    encoders["day_of_week"].classes_
)

follower_count = st.number_input(
    "Follower Count",
    min_value=0
)

likes = st.number_input(
    "Likes",
    min_value=0
)

comments = st.number_input(
    "Comments",
    min_value=0
)

shares = st.number_input(
    "Shares",
    min_value=0
)

saves = st.number_input(
    "Saves",
    min_value=0
)

reach = st.number_input(
    "Reach",
    min_value=0
)

impressions = st.number_input(
    "Impressions",
    min_value=0
)

engagement_rate = st.number_input(
    "Engagement Rate",
    min_value=0.0,
    format="%.2f"
)

followers_gained = st.number_input(
    "Followers Gained",
    min_value=0
)

caption_length = st.number_input(
    "Caption Length",
    min_value=0
)

hashtags_count = st.number_input(
    "Hashtags Count",
    min_value=0
)

# Step 7: Encode Inputs

account_type_encoded = encoders["account_type"].transform([account_type])[0]
media_type_encoded = encoders["media_type"].transform([media_type])[0]
content_category_encoded = encoders["content_category"].transform([content_category])[0]
traffic_source_encoded = encoders["traffic_source"].transform([traffic_source])[0]
day_of_week_encoded = encoders["day_of_week"].transform([day_of_week])[0]

st.markdown("---")

# Step 8: Prediction

if st.button("🚀 Predict Performance"):

    input_data = [[
        account_type_encoded,
        follower_count,
        media_type_encoded,
        content_category_encoded,
        traffic_source_encoded,
        has_call_to_action,
        post_hour,
        day_of_week_encoded,
        likes,
        comments,
        shares,
        saves,
        reach,
        impressions,
        engagement_rate,
        followers_gained,
        caption_length,
        hashtags_count
    ]]

    # Predict Class
    prediction = model.predict(input_data)[0]

    # Convert Number to Label
    prediction_label = encoders[
        "performance_bucket_label"
    ].inverse_transform([prediction])[0]

    # Prediction Probability
    probabilities = model.predict_proba(input_data)[0]

    confidence = max(probabilities) * 100
    # Display Result

    st.success(
        f"🎯 Predicted Performance: **{prediction_label.upper()}**"
    )

    st.info(
        f"📊 Confidence Score: **{confidence:.2f}%**"
    )
    # Probability Chart

    classes = encoders["performance_bucket_label"].classes_

    prob_df = pd.DataFrame({
        "Performance": classes,
        "Probability": probabilities
    })

    st.subheader("📈 Prediction Probability")

    st.bar_chart(
        prob_df.set_index("Performance")
    )

# Footer
st.markdown("---")

st.caption(
    "Developed using Python, Streamlit, Scikit-Learn and Random Forest Classifier."
)