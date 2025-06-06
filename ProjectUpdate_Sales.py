import streamlit as st
import pandas as pd
from io import StringIO

def extract_project_updates(uploaded_file):
    df = pd.read_csv(uploaded_file, encoding='utf-8')
    
    # Check if the required columns are present in the DataFrame
    required_columns = ['Project_Name', 'Project_Description', 'Acheivements_ValueAdds', 'Value_Add']
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
        <div style="background-color:#F6F5F5; padding:15px; border-radius:8px; margin-bottom:15px; border-left:4px solid #007698;">
            <h3 style="color:#021A2A; margin-top:0; font-size:20px;">{row['Project_Name']}</h3>
            <p><strong style="color:#0095D3;">Project Description:</strong> {row['Project_Description']}</p>
            <br>
            <p><strong style="color:#44D7F4;">Achievements/Value Adds:</strong></p>
            <ul style="color:#333; margin-left:20px;">
                <li>{achievements}</li>
            </ul>
            <br>
            <p><strong style="color:#F9671D;">Value Add:</strong></p>
            <ul style="color:#333; margin-left:20px;">
                <li>{value_add}</li>
            </ul>
        </div>
        """
        formatted_updates.append(formatted_update)
    
    return formatted_updates

# Streamlit App
st.set_page_config(page_title="Project Update for Sales", page_icon="📊")
st.header("📊 Project Update for Sales")
st.subheader('This Application helps you to summarise project updates from uploaded documents')

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
