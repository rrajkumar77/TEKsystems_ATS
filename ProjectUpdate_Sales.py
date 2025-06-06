import streamlit as st
import pandas as pd
from io import StringIO

def extract_project_updates(uploaded_file):
    df = pd.read_csv(uploaded_file, encoding='utf-8')
    
    # Check if the required columns are present in the DataFrame
    required_columns = ['Project_Name', 'Project_Description', 'Project_Brief_Skill', 'Acheivements_ValueAdds', 'Value_Add']
    for col in required_columns:
        if col not in df.columns:
            st.error(f"Column '{col}' is missing from the uploaded file.")
            return []
    
    project_updates = df[required_columns]
    
    formatted_updates = []
    for index, row in project_updates.iterrows():
        # Clean and format the data
        project_name = str(row['Project_Name']) if pd.notnull(row['Project_Name']) else 'N/A'
        goals = str(row['Project_Description']) if pd.notnull(row['Project_Description']) else 'N/A'
        tech_skills = str(row['Project_Brief_Skill']) if pd.notnull(row['Project_Brief_Skill']) else 'N/A'
        achievements = str(row['Acheivements_ValueAdds']) if pd.notnull(row['Acheivements_ValueAdds']) else 'N/A'
        value_delivered = str(row['Value_Add']) if pd.notnull(row['Value_Add']) else 'N/A'
        
        formatted_update = f"""
        <div style="background-color:#F6F5F5; padding:15px; border-radius:8px; margin-bottom:15px; border-left:4px solid #007698;">
            <ul style="list-style-type:none; padding-left:0; margin:0;">
                <li><strong style="color:#021A2A;">Project Name:</strong> {project_name}</li>
                <li><strong style="color:#0095D3;">Goals:</strong> {goals}</li>
                <li><strong style="color:#CDDC00;">Tech Skills:</strong> {tech_skills}</li>
                <li><strong style="color:#44D7F4;">Key Achievements:</strong> {achievements}</li>
                <li><strong style="color:#F9671D;">Value Delivered:</strong> {value_delivered}</li>
            </ul>
        </div>
        """
        formatted_updates.append(formatted_update)
    
    return formatted_updates

# Streamlit App
st.set_page_config(page_title="Project Update Analyzer", page_icon="📊")
st.header("📊 Project Update Analyzer")
st.subheader('This Application helps you to analyze project updates from uploaded documents')

# Add instructions for PowerPoint copying
st.info("💡 **Tip for PowerPoint:** After processing, you can copy the formatted project updates below and paste them directly into your PowerPoint slides.")

uploaded_file = st.file_uploader("Upload your Document (CSV only)...", type=["csv"])

if uploaded_file is not None:
    st.success("✅ Document Uploaded Successfully")
    project_updates = extract_project_updates(uploaded_file)
    
    if project_updates:
        st.subheader("📋 Project Updates")
        st.markdown("---")
        
        # Add a copy-friendly section
        st.markdown("**Instructions:** Select and copy the content below for your PowerPoint presentation:")
        
        for i, update in enumerate(project_updates, 1):
            st.markdown(f"### Project Update {i}")
            st.markdown(update, unsafe_allow_html=True)
            st.markdown("---")
    else:
        st.error("❌ No project updates could be extracted. Please check your CSV file format.")
else:
    st.markdown("👆 Please upload a CSV file to get started.")
