
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FEATURE_MODEL_PATH = BASE_DIR / "mushroom_feature_model.joblib"
IMAGE_MODEL_PATH = BASE_DIR / "mushroom_image_model.keras"
METADATA_PATH = BASE_DIR / "metadata.json"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MycoLens | Mushroom Classification",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PROFESSIONAL LIGHT UI
# ============================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: Arial, Helvetica, sans-serif;
}

.stApp {
    background: #faf8f4;
    color: #111111;
}

.main .block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main headings */

h1, h2, h3, h4, h5, h6 {
    color: #111111 !important;
    letter-spacing: -0.02em;
}

h1 {
    font-size: 2.7rem !important;
    font-weight: 750 !important;
}

h2 {
    font-size: 1.8rem !important;
}

h3 {
    font-size: 1.25rem !important;
}

/* Normal text */

p, li, label {
    color: #222222 !important;
}


/* ============================================================
   BRAND HEADER
   ============================================================ */

.brand-wrapper {
    background: #ffffff;
    border: 1px solid #e5dfd6;
    border-radius: 18px;
    padding: 24px 28px;
    margin-bottom: 22px;
    box-shadow: 0 3px 15px rgba(40, 30, 20, 0.05);
}

.brand-title {
    font-size: 2.3rem;
    font-weight: 800;
    color: #17251b;
    margin-bottom: 4px;
}

.brand-subtitle {
    font-size: 1rem;
    color: #666666;
    margin: 0;
}


/* ============================================================
   SAFETY NOTICE
   ============================================================ */

.safety-box {
    background: #245c3a;
    color: #ffffff;
    border-radius: 14px;
    padding: 17px 20px;
    margin: 18px 0 25px 0;
    border-left: 5px solid #163c26;
    box-shadow: 0 3px 12px rgba(20, 60, 35, 0.12);
}

.safety-title {
    color: #ffffff !important;
    font-size: 1rem;
    font-weight: 750;
    margin-bottom: 5px;
}

.safety-text {
    color: #f5faf6 !important;
    font-size: 0.92rem;
    line-height: 1.55;
    margin: 0;
}


/* ============================================================
   NAVIGATION
   ============================================================ */

div[data-testid="stRadio"] > label {
    display: none;
}

div[data-testid="stRadio"] > div {
    gap: 8px;
    background: #eee9e1;
    padding: 6px;
    border-radius: 13px;
    border: 1px solid #ded6cb;
    margin-bottom: 25px;
}

div[data-testid="stRadio"] label {
    background: transparent;
    border-radius: 9px;
    padding: 9px 17px;
    transition: 0.15s;
}

div[data-testid="stRadio"] label:hover {
    background: #ffffff;
}

div[data-testid="stRadio"] label p {
    font-weight: 650;
    color: #333333 !important;
}


/* ============================================================
   CARDS
   ============================================================ */

.card {
    background: #ffffff;
    border: 1px solid #e5dfd6;
    border-radius: 16px;
    padding: 23px;
    margin: 10px 0;
    box-shadow: 0 3px 14px rgba(40, 30, 20, 0.045);
}

.card-title {
    font-size: 1.1rem;
    font-weight: 750;
    color: #17251b;
    margin-bottom: 5px;
}

.card-text {
    color: #666666;
    line-height: 1.55;
}


/* ============================================================
   FEATURE INPUTS
   ============================================================ */

div[data-baseweb="select"] > div {
    background: #ffffff;
    border: 1px solid #d8d0c5;
    border-radius: 9px;
    min-height: 42px;
}

div[data-baseweb="select"] span {
    color: #111111 !important;
}

.stSelectbox label {
    color: #222222 !important;
    font-size: 0.88rem;
    font-weight: 650;
}


/* ============================================================
   BUTTONS
   ============================================================ */

.stButton > button {
    width: 100%;
    background: #315f42;
    color: #ffffff !important;
    border: none;
    border-radius: 10px;
    min-height: 46px;
    font-weight: 700;
    font-size: 0.95rem;
    transition: 0.15s;
}

.stButton > button:hover {
    background: #234b31;
    color: #ffffff !important;
    border: none;
}


/* ============================================================
   FILE UPLOADER
   ============================================================ */

div[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1px dashed #b9ad9e;
    border-radius: 14px;
    padding: 12px;
}


/* ============================================================
   METRICS
   ============================================================ */

div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e5dfd6;
    border-radius: 13px;
    padding: 16px;
    box-shadow: 0 3px 12px rgba(40, 30, 20, 0.04);
}

div[data-testid="stMetricLabel"] {
    color: #666666 !important;
}

div[data-testid="stMetricValue"] {
    color: #17251b !important;
    font-weight: 750;
}


/* ============================================================
   RESULTS
   ============================================================ */

.result-edible {
    background: #edf7ef;
    border: 1px solid #b9ddc1;
    border-radius: 14px;
    padding: 22px;
    text-align: center;
    margin: 15px 0;
}

.result-poisonous {
    background: #fff1ed;
    border: 1px solid #e7b9ad;
    border-radius: 14px;
    padding: 22px;
    text-align: center;
    margin: 15px 0;
}

.result-label {
    font-size: 1.45rem;
    font-weight: 800;
    color: #17251b;
}

.result-small {
    color: #555555;
    margin-top: 5px;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    text-align: center;
    color: #777777;
    font-size: 0.82rem;
    padding-top: 30px;
    border-top: 1px solid #e5dfd6;
    margin-top: 45px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_feature_model():
    return joblib.load(FEATURE_MODEL_PATH)


@st.cache_resource
def load_image_model():
    return tf.keras.models.load_model(IMAGE_MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r") as f:
        return json.load(f)


feature_model = load_feature_model()
image_model = load_image_model()
metadata = load_metadata()

feature_columns = metadata["feature_columns"]
category_options = metadata["category_options"]

target_mapping = metadata.get(
    "target_mapping",
    {
        "e": "Edible",
        "p": "Poisonous"
    }
)


# ============================================================
# HUMAN-READABLE FEATURE NAMES
# ============================================================

FEATURE_LABELS = {
    "cap-shape": "Cap Shape",
    "cap-surface": "Cap Surface",
    "cap-color": "Cap Color",
    "bruises": "Bruises",
    "odor": "Odor",
    "gill-attachment": "Gill Attachment",
    "gill-spacing": "Gill Spacing",
    "gill-size": "Gill Size",
    "gill-color": "Gill Color",
    "stalk-shape": "Stalk Shape",
    "stalk-root": "Stalk Root",
    "stalk-surface-above-ring": "Stalk Surface Above Ring",
    "stalk-surface-below-ring": "Stalk Surface Below Ring",
    "stalk-color-above-ring": "Stalk Color Above Ring",
    "stalk-color-below-ring": "Stalk Color Below Ring",
    "veil-type": "Veil Type",
    "veil-color": "Veil Color",
    "ring-number": "Ring Number",
    "ring-type": "Ring Type",
    "spore-print-color": "Spore Print Color",
    "population": "Population",
    "habitat": "Habitat",
}


# ============================================================
# DATASET VALUE MAPPINGS
# ============================================================

VALUE_LABELS = {

    "cap-shape": {
        "b": "Bell",
        "c": "Conical",
        "f": "Flat",
        "k": "Knobbed",
        "s": "Sunken",
        "x": "Convex",
    },

    "cap-surface": {
        "f": "Fibrous",
        "g": "Grooves",
        "s": "Scaly",
        "y": "Smooth",
    },

    "cap-color": {
        "b": "Buff",
        "c": "Cinnamon",
        "e": "Red",
        "g": "Gray",
        "n": "Brown",
        "p": "Pink",
        "r": "Green",
        "u": "Purple",
        "w": "White",
        "y": "Yellow",
    },

    "bruises": {
        "f": "No",
        "t": "Yes",
    },

    "odor": {
        "a": "Almond",
        "c": "Creosote",
        "f": "Foul",
        "l": "Anise",
        "m": "Musty",
        "n": "None",
        "p": "Pungent",
        "s": "Spicy",
        "y": "Fishy",
    },

    "gill-attachment": {
        "a": "Attached",
        "f": "Free",
    },

    "gill-spacing": {
        "c": "Close",
        "w": "Crowded",
    },

    "gill-size": {
        "b": "Broad",
        "n": "Narrow",
    },

    "gill-color": {
        "b": "Buff",
        "e": "Red",
        "g": "Gray",
        "h": "Chocolate",
        "k": "Black",
        "n": "Brown",
        "o": "Orange",
        "p": "Pink",
        "r": "Green",
        "u": "Purple",
        "w": "White",
        "y": "Yellow",
    },

    "stalk-shape": {
        "e": "Enlarging",
        "t": "Tapering",
    },

    "stalk-root": {
        "?": "Missing / Unknown",
        "b": "Bulbous",
        "c": "Club",
        "e": "Equal",
        "r": "Rooted",
    },

    "stalk-surface-above-ring": {
        "f": "Fibrous",
        "k": "Silky",
        "s": "Scaly",
        "y": "Smooth",
    },

    "stalk-surface-below-ring": {
        "f": "Fibrous",
        "k": "Silky",
        "s": "Scaly",
        "y": "Smooth",
    },

    "stalk-color-above-ring": {
        "b": "Buff",
        "c": "Cinnamon",
        "e": "Red",
        "g": "Gray",
        "n": "Brown",
        "o": "Orange",
        "p": "Pink",
        "w": "White",
        "y": "Yellow",
    },

    "stalk-color-below-ring": {
        "b": "Buff",
        "c": "Cinnamon",
        "e": "Red",
        "g": "Gray",
        "n": "Brown",
        "o": "Orange",
        "p": "Pink",
        "w": "White",
        "y": "Yellow",
    },

    "veil-type": {
        "p": "Partial",
    },

    "veil-color": {
        "n": "Brown",
        "o": "Orange",
        "w": "White",
        "y": "Yellow",
    },

    "ring-number": {
        "n": "None",
        "o": "One",
        "t": "Two",
    },

    "ring-type": {
        "e": "Evanescent",
        "f": "Flaring",
        "l": "Large",
        "n": "None",
        "p": "Pendant",
    },

    "spore-print-color": {
        "b": "Buff",
        "h": "Chocolate",
        "k": "Black",
        "n": "Brown",
        "o": "Orange",
        "r": "Green",
        "u": "Purple",
        "w": "White",
        "y": "Yellow",
    },

    "population": {
        "a": "Abundant",
        "c": "Clustered",
        "n": "Numerous",
        "s": "Scattered",
        "v": "Several",
        "y": "Solitary",
    },

    "habitat": {
        "d": "Woods",
        "g": "Grasses",
        "l": "Leaves",
        "m": "Meadows",
        "p": "Paths",
        "u": "Urban",
        "w": "Waste",
    },
}


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="brand-wrapper">
    <div class="brand-title">🍄 MycoLens</div>
    <p class="brand-subtitle">
        Mushroom Classification & Machine Learning Analysis
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SAFETY NOTICE
# ============================================================

st.markdown("""
<div class="safety-box">
    <div class="safety-title">Important Safety Note</div>
    <p class="safety-text">
        This application is for educational and machine-learning
        demonstration purposes only. Model predictions should never
        be used as a real-world method for determining whether a
        wild mushroom is safe to eat.
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# NAVIGATION
# ============================================================

page = st.radio(
    "Navigation",
    [
        "Feature Analysis",
        "Image Analysis",
        "Model Performance",
        "About"
    ],
    horizontal=True
)


# ============================================================
# FEATURE ANALYSIS
# ============================================================

if page == "Feature Analysis":

    st.header("Feature Analysis")

    st.markdown("""
    <div class="card">
        <div class="card-title">Classify using mushroom characteristics</div>
        <div class="card-text">
            Select the observed characteristics below. The application
            displays understandable descriptions while internally using
            the original dataset codes expected by the trained model.
        </div>
    </div>
    """, unsafe_allow_html=True)

    user_input = {}

    col1, col2 = st.columns(2, gap="large")

    for i, feature in enumerate(feature_columns):

        codes = category_options.get(feature, [])

        if not codes:
            st.error(
                f"No category information found for {feature}."
            )
            continue

        readable_options = [
            VALUE_LABELS.get(feature, {}).get(code, code)
            for code in codes
        ]

        display_to_code = {
            VALUE_LABELS.get(feature, {}).get(code, code): code
            for code in codes
        }

        with col1 if i % 2 == 0 else col2:

            selected_readable = st.selectbox(
                FEATURE_LABELS.get(feature, feature),
                readable_options,
                key=f"feature_{feature}"
            )

            user_input[feature] = display_to_code[selected_readable]

    st.write("")

    if st.button(
        "🔍  Analyze Mushroom",
        use_container_width=True
    ):

        input_df = pd.DataFrame(
            [user_input],
            columns=feature_columns
        )

        prediction = feature_model.predict(input_df)[0]
        predicted_label = str(prediction)

        confidence = None

        if hasattr(feature_model, "predict_proba"):

            probabilities = feature_model.predict_proba(input_df)[0]
            classes = list(feature_model.classes_)

            probability_dict = dict(
                zip(classes, probabilities)
            )

            confidence = probability_dict.get(
                predicted_label,
                max(probabilities)
            )

        st.divider()

        st.subheader("Classification Result")

        if predicted_label.lower() == "poisonous":

            st.markdown("""
            <div class="result-poisonous">
                <div class="result-label">Model Prediction: Poisonous</div>
                <div class="result-small">
                    Based on the selected dataset characteristics.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="result-edible">
                <div class="result-label">Model Prediction: Edible</div>
                <div class="result-small">
                    Based on the selected dataset characteristics.
                </div>
            </div>
            """, unsafe_allow_html=True)

        if confidence is not None:

            st.metric(
                "Model Confidence",
                f"{confidence * 100:.2f}%"
            )

        st.caption(
            "This confidence value represents the model's output for "
            "this dataset-based prediction. It is not a measure of "
            "real-world mushroom identification or food safety."
        )


# ============================================================
# IMAGE ANALYSIS
# ============================================================

elif page == "Image Analysis":

    st.header("Image Analysis")

    st.markdown("""
    <div class="card">
        <div class="card-title">Classify a mushroom image</div>
        <div class="card-text">
            Upload an image and the trained MobileNetV2 model will
            generate a classification prediction.
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a JPG, JPEG, or PNG image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        image_col, info_col = st.columns(
            [1.15, 0.85],
            gap="large"
        )

        with image_col:

            st.image(
                image,
                caption="Uploaded Mushroom Image",
                use_container_width=True
            )

        with info_col:

            st.markdown("""
            <div class="card">
                <div class="card-title">Image Analysis</div>
                <div class="card-text">
                    The image is resized to 224 × 224 pixels and
                    processed by the trained MobileNetV2 classifier.
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(
                "🔍  Analyze Image",
                use_container_width=True
            ):

                img = image.resize((224, 224))

                img_array = np.array(
                    img
                ).astype("float32") / 255.0

                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )

                prediction = image_model.predict(
                    img_array,
                    verbose=0
                )

                probability = float(
                    np.squeeze(prediction)
                )

                if probability >= 0.5:

                    predicted_class = "Poisonous"
                    confidence = probability

                else:

                    predicted_class = "Edible"
                    confidence = 1 - probability

                st.divider()

                if predicted_class == "Poisonous":

                    st.markdown("""
                    <div class="result-poisonous">
                        <div class="result-label">
                            Model Prediction: Poisonous
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                else:

                    st.markdown("""
                    <div class="result-edible">
                        <div class="result-label">
                            Model Prediction: Edible
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.metric(
                    "Model Confidence",
                    f"{confidence * 100:.2f}%"
                )

                st.caption(
                    "The image model achieved 94.12% validation accuracy "
                    "on the project's validation dataset. Individual "
                    "predictions can still be incorrect."
                )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header("Model Performance")

    st.markdown("""
    <div class="card">
        <div class="card-title">Feature-based model comparison</div>
        <div class="card-text">
            The following results were obtained during evaluation
            of the structured mushroom dataset.
        </div>
    </div>
    """, unsafe_allow_html=True)

    performance_df = pd.DataFrame({
        "Model": [
            "Extra Trees",
            "Random Forest",
            "Linear SVM",
            "Logistic Regression"
        ],
        "5-Fold CV Accuracy": [
            1.000000,
            1.000000,
            0.999692,
            0.999385
        ],
        "Test Accuracy": [
            1.000000,
            1.000000,
            1.000000,
            0.999385
        ]
    })

    st.dataframe(
        performance_df,
        use_container_width=True,
        hide_index=True
    )

    st.write("")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "Selected Model",
            "Extra Trees"
        )

    with metric2:
        st.metric(
            "Feature Test Accuracy",
            "100%"
        )

    with metric3:
        st.metric(
            "Image Validation Accuracy",
            "94.12%"
        )

    st.subheader("Image Model")

    st.write(
        "The image classifier uses MobileNetV2 and achieved 94.12% "
        "accuracy on the project's labelled validation image dataset."
    )

    st.caption(
        "These are dataset-specific benchmark results. They do not "
        "represent guaranteed real-world performance."
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    st.header("About MycoLens")

    st.markdown("""
    <div class="card">
        <div class="card-title">Project Overview</div>
        <div class="card-text">
            MycoLens is a student machine-learning project exploring
            mushroom classification through both structured
            characteristics and image analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Structured Classification")

    st.write(
        "The feature-based model uses an Extra Trees classifier "
        "with categorical preprocessing and 22 mushroom characteristics."
    )

    st.subheader("Image Classification")

    st.write(
        "The image-based model uses MobileNetV2 and was evaluated "
        "using a separate labelled image validation dataset."
    )

    st.subheader("Dataset")

    st.write(
        "The structured dataset contains 8,124 records and 22 "
        "categorical input features with edible and poisonous labels."
    )

    st.subheader("Project Results")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Feature Model",
            "Extra Trees"
        )
        st.metric(
            "Feature Test Accuracy",
            "100%"
        )

    with result_col2:
        st.metric(
            "Image Model",
            "MobileNetV2"
        )
        st.metric(
            "Image Validation Accuracy",
            "94.12%"
        )

    st.subheader("Safety")

    st.markdown("""
    <div class="safety-box">
        <div class="safety-title">Educational Use Only</div>
        <p class="safety-text">
            This application demonstrates machine-learning techniques
            and should not be used to determine whether a wild mushroom
            is safe to eat. Never rely on an AI prediction for
            mushroom consumption decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    MycoLens · Machine Learning Mushroom Classification Project
</div>
""", unsafe_allow_html=True)
