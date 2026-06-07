import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ====================================================
# PAGE CONFIG
# ====================================================

st.set_page_config(
    page_title="Titanic Analytics + Prediction",
    layout="wide"
)

st.title("🚢 Titanic Analytics + ML Chatbot")

# ====================================================
# LOAD DATA
# ====================================================

@st.cache_data
def load_data():
    return pd.read_excel("titanic_excel.xlsx")

df = load_data()

# ====================================================
# TRAIN MODEL
# ====================================================

@st.cache_resource
def train_model(df):

    ml_df = df.copy()

    target = "Survived"

    cols_to_drop = ["Name", "Ticket", "Cabin"]

    existing_cols = [
        col for col in cols_to_drop
        if col in ml_df.columns
    ]

    ml_df = ml_df.drop(
        columns=existing_cols
    )

    encoders = {}

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

    X = ml_df.drop(columns=[target])
    y = ml_df[target]

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

    return model, X, accuracy, encoders

model, X, accuracy, encoders = train_model(df)

# ====================================================
# GROQ CLIENT
# ====================================================

client = Groq(
    api_key=st.secrets["key"]
)

# ====================================================
# CHATBOT FUNCTION
# ====================================================

def ask_data(question):

    q = question.lower()

    # ------------------------------
    # Accuracy
    # ------------------------------

    if "accuracy" in q:
        return f"Model Accuracy = {accuracy:.2f}"

    # ------------------------------
    # Feature Importance Graph
    # ------------------------------

    if "feature importance" in q:

        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(
            "Importance",
            ascending=False
        )

        fig = px.bar(
            importance,
            x="Feature",
            y="Importance",
            title="Feature Importance"
        )

        return fig

    # ------------------------------
    # Predict All Records
    # ------------------------------

    if "predict all" in q:

        preds = model.predict(X)

        result = pd.DataFrame({
            "Prediction": preds
        })

        return result

    # ------------------------------
    # LLM Prompt
    # ------------------------------

    prompt = f"""
You are an expert Python Data Analyst.

Dataset columns:
{list(df.columns)}

Dataframe name: df

Machine Learning Features:
{list(X.columns)}

Rules:

1. Return ONLY executable Python code.
2. Use dataframe name df.
3. Use model name model.
4. Use plotly.express as px for charts.
5. Store final output in variable result.
6. No explanation.
7. No markdown.
8. If chart requested return Plotly figure.

Examples:

Histogram:
result = px.histogram(df,x='Age')

Bar Chart:
result = px.bar(df,x='Sex',y='Fare')

Scatter:
result = px.scatter(df,x='Age',y='Fare',color='Survived')

Box:
result = px.box(df,x='Pclass',y='Fare')

Pie:
result = px.pie(df,names='Sex')

Correlation:
Correlation=df.select_dtypes(include='number').corr()
result=px.imshow(corr,text_auto=True)

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
            "px": px,
            "model": model,
            "X": X
        }

        exec(code, {}, local_vars)

        return local_vars.get(
            "result",
            "No result returned."
        )

    except Exception as e:

        return f"""
Error:
{e}

Generated Code:
{code}
"""

# ====================================================
# SIDEBAR
# ====================================================

page = st.sidebar.radio(
    "Choose Option",
    [
        "Prediction",
        "Analytics Chatbot",
        "Dataset"
    ]
)

# ====================================================
# DATASET PAGE
# ====================================================

if page == "Dataset":

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.metric(
        "Model Accuracy",
        f"{accuracy:.2f}"
    )

# ====================================================
# PREDICTION PAGE
# ====================================================

elif page == "Prediction":

    st.subheader(
        "Passenger Survival Prediction"
    )

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

                median_value = float(
                    pd.to_numeric(
                        df[col],
                        errors="coerce"
                    ).median()
                )

                user_input[col] = st.number_input(
                    col,
                    value=median_value
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
                "Passenger SURVIVED ✅"
            )

        else:

            st.error(
                "Passenger DID NOT SURVIVE ❌"
            )

        st.write(
            f"Survival Probability: {probability[1]:.2%}"
        )

# ====================================================
# CHATBOT PAGE
# ====================================================

elif page == "Analytics Chatbot":

    st.subheader(
        "Ask Questions About Titanic Data"
    )

    question = st.text_input(
        "Enter your question"
    )

    if st.button("Ask"):

        answer = ask_data(question)

        st.write("### Answer")

        if isinstance(
            answer,
            pd.DataFrame
        ):

            st.dataframe(answer)

        elif isinstance(
            answer,
            pd.Series
        ):

            st.dataframe(answer)

        elif isinstance(
            answer,
            go.Figure
        ):

            st.plotly_chart(
                answer,
                use_container_width=True
            )

        else:

            st.write(answer)