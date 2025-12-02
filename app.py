import streamlit as st
import pickle
import pandas as pd
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier  # just to help unpickling

st.set_page_config(page_title="Hotel Booking Status Prediction", page_icon="🏨")

# ---------- Paths ----------
HERE = Path(__file__).parent
MODEL_PATH = HERE / "decision_tree_model.pkl"
DATA_INFO_PATH = HERE / "data_info.pkl"

# ---------- Helper to load pickles ----------
@st.cache_resource
def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)

# ---------- Load model ----------
try:
    model = load_pickle(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load model at {MODEL_PATH}.\n\n{e}")
    st.stop()

# ---------- Load data_info ----------
try:
    data_info = load_pickle(DATA_INFO_PATH)
except Exception as e:
    st.error(
        f"Could not load data_info at {DATA_INFO_PATH}.\n"
        "Ensure data_info.pkl exists and was created from the same training code.\n\n"
        f"{e}"
    )
    st.stop()

expected_columns = data_info["expected_columns"]
feature_order = data_info["feature_order"]
categorical_unique_values = data_info["categorical_unique_values"]
numeric_ranges = data_info.get("numeric_ranges", {})
ohe_categorical_columns = data_info["ohe_categorical_columns"]
numeric_columns = data_info["numeric_columns"]
target_classes = data_info["target_classes"]

# ---------- Streamlit UI ----------
st.title("Hotel Booking Status Prediction")
st.caption("This app uses your trained Decision Tree model on the hotel reservations dataset.")

st.header("Enter Booking Details")

# Helper for numeric sliders
def numeric_input_slider(col_name: str):
    rng = numeric_ranges.get(col_name, {})
    lo = rng.get("min", 0.0)
    hi = rng.get("max", 100.0)
    default = rng.get("default", (lo + hi) / 2)

    # Try to choose reasonable step size
    step = max((hi - lo) / 100, 1.0)

    return st.slider(
        label=col_name.replace("_", " ").title(),
        min_value=float(lo),
        max_value=float(hi),
        value=float(default),
        step=float(step),
    )

# Prepare a dictionary to hold user inputs (original feature space, before encoding)
user_inputs = {}

# Go through features in the original order
for col in feature_order:
    if col in numeric_columns:
        user_inputs[col] = numeric_input_slider(col)
    elif col in ohe_categorical_columns:
        options = categorical_unique_values.get(col, [])
        if not options:
            # fallback if something goes wrong
            user_inputs[col] = st.text_input(col, "")
        else:
            user_inputs[col] = st.selectbox(
                col.replace("_", " ").title(),
                options
            )
    else:
        # If a column sneaks in that is neither numeric nor categorical,
        # just let the user type something
        user_inputs[col] = st.text_input(col, "")

st.divider()

if st.button("Predict Booking Status"):
    # ---------- Build a single-row DataFrame ----------
    raw_df = pd.DataFrame([user_inputs])

    # ---------- Apply same one-hot encoding as training ----------
    input_encoded = pd.get_dummies(
        raw_df,
        columns=ohe_categorical_columns,
        drop_first=True,
        dtype=int
    )

    # Add any missing columns that were in training
    for col in expected_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Make sure column order matches training
    input_encoded = input_encoded[expected_columns]

    # ---------- Run the model ----------
    try:
        pred_int = model.predict(input_encoded)[0]
        proba_fn = getattr(model, "predict_proba", None)

        # Map integer prediction back to original label
        if 0 <= pred_int < len(target_classes):
            pred_label = target_classes[pred_int]
        else:
            pred_label = str(pred_int)

        st.subheader("Prediction Result")
        st.write(f"**Predicted Booking Status:** {pred_label}")

        if callable(proba_fn):
            probs = proba_fn(input_encoded)[0]
            # In scikit-learn, the order of columns in predict_proba is model.classes_
            # which should align with the integer codes you used.
            st.write("**Prediction Probabilities:**")
            for class_idx, class_label in enumerate(target_classes):
                st.write(f"- {class_label}: {probs[class_idx]:.3f}")

    except Exception as e:
        st.error(f"Prediction failed:\n\n{e}")
