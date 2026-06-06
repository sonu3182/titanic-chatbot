import streamlit as st
import pandas as pd
import numpy as np
from groq import Groq
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Titanic Analytics & Prediction",
    layout="wide"
)

st.title("🚢 Titanic Analytics + ML Chatbot")

# =====================================
# LOAD DATA
# =====================================

@st.cache_data
def load_data():
    return pd.read_excel("titanic_excel.xlsx")

df = load_data()

# =====================================
# TRAIN MODEL
# =====================================

@st.cache_resource
def train_model(df):

    ml_df = df.copy()

    TARGET = "Survived"

    # Drop columns that don't help prediction
    cols_to_drop = ["Name", "Ticket", "Cabin"]

    existing_cols = [
        col for col in cols_to_drop
        if col in ml_df.columns
    ]

    ml_df = ml_df.drop(
        columns=existing_cols
    )

    encoders = {}

    # Numeric columns
    numeric_cols = [
        "PassengerId",
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare"
    ]

    for col in numeric_cols:

        if col in ml_df.columns:

            ml_df[col] = pd.to_numeric(
                ml_df[col],
                errors="coerce"
            )

            ml_df[col] = ml_df[col].fillna(
                ml_df[col].median()
            )

    # Categorical columns
    categorical_cols = [
        "Sex",
        "Embarked"
    ]

    for col in categorical_cols:

        if col in ml_df.columns:

            ml_df[col] = (
                ml_df[col]
                .fillna("Unknown")
                .astype(str)
            )

            le = LabelEncoder()

            ml_df[col] = le.fit_transform(
                ml_df[col]
            )

            encoders[col] = le

    X = ml_df.drop(columns=[TARGET])

    y = ml_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    accuracy = model.score(
        X_test,
        y_test
    )

    return (
        model,
        X,
        accuracy,
        encoders,
        ml_df
    )
model, X, accuracy, encoders, ml_df = train_model(df)

# =====================================
# GROQ CLIENT
# =====================================

client = Groq(
    api_key=st.secrets['key']
)

# =====================================
# CHATBOT FUNCTION
# =====================================

def ask_data(question):

    q = question.lower()

    if "accuracy" in q:
        return f"Model Accuracy = {accuracy:.2f}"

    if "feature importance" in q:

        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        })

        return importance.sort_values(
            "Importance",
            ascending=False
        )

    if "predict all" in q:

        preds = model.predict(X)

        result = pd.DataFrame({
            "Prediction": preds
        })

        return result.head(20)

    prompt = f"""
You are an Expert Data Analyst.

Dataset Columns:
{list(df.columns)}

Dataset Shape:
{df.shape}

Target Column:
Survived

Generate ONLY executable Python code.

Rules:
1. DataFrame name is df
2. ML Features dataframe name is X
3. ML Model name is model
4. Store final answer in variable named result
5. No explanation
6. Return only code

Question:
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    code = response.choices[0].message.content

    code = (
        code.replace("```python", "")
        .replace("```", "")
        .strip()
    )

    try:

        local_vars = {
            "df": df,
            "pd": pd,
            "np": np,
            "X": X,
            "model": model
        }

        exec(code, {}, local_vars)

        return local_vars.get(
            "result",
            "No result returned"
        )

    except Exception as e:

        return f"""
Error: {e}

Generated Code:
{code}
"""

# =====================================
# SIDEBAR
# =====================================

page = st.sidebar.radio(
    "Choose Option",
    [
        "Prediction",
        "Analytics Chatbot",
        "Dataset"
    ]
)

# =====================================
# DATASET PAGE
# =====================================

if page == "Dataset":

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.metric(
        "Model Accuracy",
        f"{accuracy:.2f}"
    )

# =====================================
# PREDICTION PAGE
# =====================================

elif page == "Prediction":

    st.subheader("Passenger Survival Prediction")

    user_input = {}

    cols = st.columns(2)

    for i, col in enumerate(X.columns):

        with cols[i % 2]:

            if col in encoders:

                options = list(
                    encoders[col].classes_
                )

                user_input[col] = st.selectbox(
                    col,
                    options
                )

            else:

                default_value = float(
                        pd.to_numeric(
                            df[col],
                            errors="coerce"
                        ).median()
                    )

                user_input[col] = st.number_input(
                    col,
                    value=default_value
                )

    if st.button("Predict Survival"):

        input_df = pd.DataFrame(
            [user_input]
        )

        for col, encoder in encoders.items():

            if col in input_df.columns:

                input_df[col] = encoder.transform(
                    input_df[col].astype(str)
                )

        prediction = model.predict(
            input_df
        )[0]

        probability = model.predict_proba(
            input_df
        )[0]

        if prediction == 1:

            st.success(
                f"Passenger SURVIVED ✅"
            )

        else:

            st.error(
                f"Passenger DID NOT SURVIVE ❌"
            )

        st.write(
            f"Survival Probability: {probability[1]:.2%}"
        )

# =====================================
# CHATBOT PAGE
# =====================================

elif page == "Analytics Chatbot":

    st.subheader("Ask Questions About Data")

    question = st.text_input(
        "Enter your question"
    )

    if st.button("Ask"):

        answer = ask_data(question)

        st.write("### Answer")

        if isinstance(
            answer,
            (pd.DataFrame, pd.Series)
        ):
            st.dataframe(answer)
        else:
            st.write(answer)