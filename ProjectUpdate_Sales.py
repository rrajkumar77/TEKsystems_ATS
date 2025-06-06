import streamlit as st
import pandas as pd
from io import StringIO

def extract_project_updates(uploaded_file):
    df = pd.read_csv(uploaded_file, encoding='utf-8')
    
    # Check if the required columns are present in the DataFrame
    required_columns = ['Team_Lead', 'Project_Name', 'Project_Description', 'Acheivements_ValueAdds', 'Value_Add']
    for col in required_columns:
        if col not in df.columns:
            st.error(f"Column '{col}' is missing from the uploaded file.")
            return []
    
    project_updates = df[required_columns]
    
    formatted_updates = []
    for index, row in project_updates.iterrows():
        achievements = str(row['Acheivements_ValueAdds']).replace(';', '.</li>\n<li>') if pd.notnull(row['Acheivements_ValueAdds']) else ''
        value_add = str(row['Value_Add']).replace(';', '.</li>\n<li>') if pd.notnull(row['Value_Add']) else ''
        
        formatted_update = f"""
        <div style="background-color:#F6F5F5; padding:10px; border-radius:5px; margin-bottom:10px;">
            <h3 style="color:#021A2A;">{row['Project_Name']}</h3>
            <p><strong style="color:#CDDC00;">Lead Name:</strong> {row['Team_Lead']}</p>
            <br>
            <p><strong style="color:#0095D3;">Project Description:</strong> {row['Project_Description']}</p>
            <br>
            <p><strong style="color:#44D7F4;">Achievements/Value Adds:</strong></p>
            <ul style="color:#333;">
                <li>{achievements}</li>
            </ul>
            <br>
            <p><strong style="color:#F9671D;">Value Add:</strong></p>
            <ul style="color:#333;">
                <li>{value_add}</li>
            </ul>
        </div>
        """
        formatted_updates.append(formatted_update)
    
    return formatted_updates

# Streamlit App
st.set_page_config(page_title="Document Analyser")
st.header("Document Analyzer")
st.subheader('This Application helps you to Analyse any document uploaded')
uploaded_file = st.file_uploader("Upload your Document (CSV only)...", type=["csv"])
if uploaded_file is not None:
    st.write("Document Uploaded Successfully")
    project_updates = extract_project_updates(uploaded_file)
    st.subheader("Project Updates")
    for update in project_updates:
        st.markdown(update, unsafe_allow_html=True)
