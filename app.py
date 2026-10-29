import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Question Paper Difficulty Analyzer",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("difficulty_model.pkl")

model = load_model()

def prepare_text(dataframe):
    return (
        "Subject: " + dataframe["subject"].astype(str)
        + " Marks: " + dataframe["marks"].astype(str)
        + " Question: " + dataframe["question"].astype(str)
    )

def classify_questions(dataframe):
    result = dataframe.copy()
    text = prepare_text(result)

    result["Predicted Difficulty"] = model.predict(text)
    probabilities = model.predict_proba(text)
    result["Model Confidence (%)"] = (
        probabilities.max(axis=1) * 100
    ).round(1)

    return result

st.title("📚 AI-Based Question Paper Difficulty Analyzer")
st.write(
    "Analyze university-style questions from AIML, "
    "Computer Networks and Cloud Computing."
)
st.caption(
    "Prototype: predictions depend on the training data "
    "and should be reviewed by a subject expert."
)

tab1, tab2 = st.tabs([
    "✍️ Enter a Question",
    "📁 Upload Question Paper CSV"
])

with tab1:
    st.subheader("Predict a single question")

    question = st.text_area(
        "Enter your exam question",
        placeholder="Example: Explain VPC peering with a suitable diagram."
    )

    subject = st.selectbox(
        "Select subject",
        ["AIML", "CN", "Cloud Computing"]
    )

    marks = st.selectbox(
        "Marks",
        [2, 5, 10]
    )

    if st.button("Analyze Question", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            single = pd.DataFrame([{
                "question": question.strip(),
                "subject": subject,
                "marks": marks
            }])

            output = classify_questions(single)
            predicted = output.loc[0, "Predicted Difficulty"]

            st.subheader("Prediction")
            if predicted == "Easy":
                st.success(f"Predicted difficulty: {predicted}")
            elif predicted == "Medium":
                st.info(f"Predicted difficulty: {predicted}")
            else:
                st.warning(f"Predicted difficulty: {predicted}")

            st.write(
                "Model confidence:",
                f"{output.loc[0, 'Model Confidence (%)']}%"
            )

with tab2:
    st.subheader("Analyze multiple questions")

    st.write(
        "Upload a CSV with columns: question, subject, marks. "
        "The app will predict difficulty for each row."
    )

    uploaded_file = st.file_uploader(
        "Upload question paper CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)

            required = {"question", "subject", "marks"}
            missing = required - set(input_df.columns)

            if missing:
                st.error(
                    "Missing required columns: "
                    + ", ".join(sorted(missing))
                )
            elif input_df.empty:
                st.warning("The uploaded CSV has no questions.")
            else:
                input_df = input_df.dropna(
                    subset=["question", "subject", "marks"]
                ).copy()

                input_df["question"] = (
                    input_df["question"].astype(str).str.strip()
                )
                input_df["subject"] = (
                    input_df["subject"].astype(str).str.strip()
                )
                input_df["marks"] = pd.to_numeric(
                    input_df["marks"], errors="coerce"
                )

                input_df = input_df.dropna(subset=["marks"])
                input_df = input_df[input_df["question"] != ""]

                if input_df.empty:
                    st.warning("No valid questions were found.")
                else:
                    result = classify_questions(input_df)

                    st.subheader("Question-wise results")
                    st.dataframe(result, use_container_width=True)

                    counts = (
                        result["Predicted Difficulty"]
                        .value_counts()
                        .reindex(
                            ["Easy", "Medium", "Hard"],
                            fill_value=0
                        )
                    )

                    st.subheader("Difficulty distribution")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Easy", int(counts["Easy"]))
                    col2.metric("Medium", int(counts["Medium"]))
                    col3.metric("Hard", int(counts["Hard"]))

                    fig, ax = plt.subplots()
                    counts.plot(kind="bar", ax=ax)
                    ax.set_title("Question Paper Difficulty")
                    ax.set_xlabel("Predicted difficulty")
                    ax.set_ylabel("Number of questions")
                    ax.tick_params(axis="x", rotation=0)
                    st.pyplot(fig)
                    plt.close(fig)

                    csv_data = result.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        "Download Analysis CSV",
                        data=csv_data,
                        file_name="difficulty_analysis.csv",
                        mime="text/csv"
                    )

        except Exception as error:
            st.error(f"Could not process this CSV: {error}")

st.divider()
st.caption(
    "AIML Mini Project | TF-IDF + Logistic Regression"
)
