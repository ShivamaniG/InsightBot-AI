import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from insightbot_pipeline import insightbot_process
from retriever import retrieve  
from summarization import summarize_texts
from pdf2image import convert_from_bytes
from PIL import Image


st.set_page_config(page_title="InsightBot", layout="wide")

# Sidebar
st.sidebar.title("📂 Document Upload & Status")
uploaded_file = st.sidebar.file_uploader("Upload PDF, TXT, DOCX or CSV", type=["pdf", "txt", "docx", "csv"])
# embed_status = st.sidebar.radio("🧠 Embedding Status", ["Not Embedded", "Embedded"])

if uploaded_file:
    file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type}
    st.sidebar.success(f"Uploaded: {file_details['filename']}")

# Layout
st.title("📊 InsightBot Dashboard")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Chat with InsightBot",
    "📊 Insight Cards",
    "📝 Summarization Panel",
    "📄 Document Preview",
    "📈 CSV Visualizations"
])

# Tab 1: Chat
with tab1:
    st.header("💬 Ask InsightBot")
    query = st.text_input("Enter your question:")

    if st.button("Ask"):
        if not uploaded_file:
            st.warning("Please upload a document first.")
        elif not query.strip():
            st.warning("Please enter a question.")
        else:
            try:
                file_bytes = uploaded_file.read()
                answer, retrieved_chunks, insights = insightbot_process(file_bytes, uploaded_file.name, query)

                st.success(f"🔍 Answer: {answer}")

                with st.expander("📁 Related Chunks"):
                    for i, chunk in enumerate(retrieved_chunks):
                        st.markdown(f"**Excerpt {i+1}:** {chunk['text']}")

            except Exception as e:
                st.error(f"Error: {e}")

with tab2:
    st.header("📊 Insight Cards")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🧠 Key Points")
        if "insights" in locals():
            for point in insights.get("Key Points", []):
                st.markdown(f"- {point}")

    with col2:
        st.markdown("### 💡 Major Insights")
        if "insights" in locals():
            for insight in insights.get("Major Insights", []):
                st.markdown(f"- {insight}")

    with col3:
        st.markdown("### ✅ Action Items")
        if "insights" in locals():
            for item in insights.get("Action Items", []):
                st.markdown(f"- {item}")


# Tab 3: Summarization
with tab3:
    st.header("📝 Summarization Panel")

    if st.button("Summarize"):
        # Use retriever to get top chunks (no input needed now)
        chunks = retrieve("", top_k=10)  # empty query or modify as needed
        summary = summarize_texts(chunks)
        st.markdown(f"**Summary:** {summary}")


# Tab 4: Document Preview
with tab4:
    st.header("📄 Document Preview")
    if uploaded_file:
        if uploaded_file.name.endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            st.markdown(f"**{uploaded_file.name} Preview:**")
            st.write(reader.pages[0].extract_text())
        elif uploaded_file.name.endswith(".txt"):
            st.text(uploaded_file.read().decode())
        elif uploaded_file.name.endswith(".docx"):
            doc = Document(uploaded_file)
            fullText = '\n'.join([para.text for para in doc.paragraphs])
            st.text(fullText[:1000])
    else:
        st.info("No document uploaded.")

# Tab 5: CSV Visualizations
with tab5:
    st.header("📈 Company CSV Visualizations")
    if uploaded_file and uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        st.subheader("🔍 Preview of Data")
        st.dataframe(df.head())

        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if numeric_cols:
            x_axis = st.selectbox("Select X-axis column", numeric_cols)
            y_axis = st.selectbox("Select Y-axis column", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

            st.subheader("📊 Bar Plot")
            fig1, ax1 = plt.subplots()
            sns.barplot(data=df, x=x_axis, y=y_axis, ax=ax1)
            st.pyplot(fig1)

            st.subheader("🎯 Pie Chart")
            pie_col = st.selectbox("Select column for Pie Chart (categorical)", df.columns)
            pie_data = df[pie_col].value_counts()
            fig2, ax2 = plt.subplots()
            ax2.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%')
            st.pyplot(fig2)
        else:
            st.warning("No numeric columns found for visualization.")
    else:
        st.info("Upload a CSV file to view visualizations.")
