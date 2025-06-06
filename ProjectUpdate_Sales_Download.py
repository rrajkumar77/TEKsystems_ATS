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
            "Include project name, goals, tech skills used, key achievements, and value delivered:"
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
    
    # Storage for all summaries
    all_summaries = []
    all_summaries_html = []

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
        
        # Store summaries for download
        all_summaries.append({
            'Project_Name': project_name,
            'Employee_Name': employee_name,
            'Team_Lead': team_lead,
            'Summary': summary_html.replace('<ul>', '').replace('</ul>', '').replace('<li>', '• ').replace('</li>', '\n')
        })
        all_summaries_html.append(formatted_output)
    
    # Download options
    st.markdown("---")
    st.subheader("📥 Download Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download as CSV
        summary_df = pd.DataFrame(all_summaries)
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📊 Download as CSV",
            data=csv,
            file_name="qbr_project_summaries.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download as HTML
        html_content = f"""
        <html>
        <head>
            <title>QBR Project Summaries</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
            </style>
        </head>
        <body>
            <h1>QBR Project Summaries</h1>
            {''.join(all_summaries_html)}
        </body>
        </html>
        """
        st.download_button(
            label="🌐 Download as HTML",
            data=html_content,
            file_name="qbr_project_summaries.html",
            mime="text/html"
        )
    
    with col3:
        # Download as Text
        text_content = ""
        for summary in all_summaries:
            text_content += f"PROJECT: {summary['Project_Name']}\n"
            text_content += f"EMPLOYEE: {summary['Employee_Name']}\n"
            text_content += f"TEAM LEAD: {summary['Team_Lead']}\n"
            text_content += f"SUMMARY:\n{summary['Summary']}\n"
            text_content += "-" * 50 + "\n\n"
        
        st.download_button(
            label="📄 Download as TXT",
            data=text_content,
            file_name="qbr_project_summaries.txt",
            mime="text/plain"
        )
