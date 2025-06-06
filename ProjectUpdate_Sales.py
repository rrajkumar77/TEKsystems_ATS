import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ GOOGLE_API_KEY not set in your .env file.")
else:
    genai.configure(api_key=api_key)

# Format summary with brand colors
def format_summary(project_name, summary_html):
    return f"""
    <div style="background-color:#F5F5F5; padding: 20px; border-radius: 12px; margin-bottom: 25px;">
        <h3 style="color:#012A52; font-family:sans-serif;">{project_name}</h3>
        <div style="color:#00798B; font-size: 16px; font-family:sans-serif;">{summary_html}</div>
    </div>
    """

# Generate summary using Gemini
def get_project_summary(text):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "Summarize this employee's QBR project in 4-5 concise bullet points using <ul><li> HTML tags. "
            "Include goals, tech skills used, key achievements, and value delivered:"
        )
        response = model.generate_content([prompt + "\n\n" + text])
        return response.text
    except Exception as e:
        return f"<p style='color:red;'>Error: {str(e)}</p>"

# Streamlit UI
st.set_page_config(page_title="QBR Summary Generator", layout="centered")
st.title("📊 QBR Project Summary Generator")

uploaded_file = st.file_uploader("Upload the QBR CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded!")

    for idx, row in df.iterrows():
        employee_name = row.get("Created By", "N/A")
        team_lead = row.get("Team_Lead", "")
        project_name = row.get("Project_Name", "")
        project_desc = row.get("Project_Description", "")
        achievements = row.get("Acheivements_ValueAdds", "")
        value_add = row.get("Value_Add", "")

        combined_text = f"""
        Title: {employee_name}
        Team Lead: {team_lead}
        Project Name: {project_name}
        Project Description: {project_desc}
        Achievements: {achievements}
        Value Add: {value_add}
        """

        summary_html = get_project_summary(combined_text)
        formatted_output = format_summary(project_name, summary_html)
        st.markdown(formatted_output, unsafe_allow_html=True)
