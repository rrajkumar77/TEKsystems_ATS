import streamlit as st
import google.generativeai as genai
import os
import re
from dotenv import load_dotenv
import docx
import pandas as pd
import fitz  # PyMuPDF

# Set page configuration at the very beginning
st.set_page_config(page_title="JD and Resume Matcher with Skills")
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input_prompt, resume_content, jd_content):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input_prompt, resume_content, jd_content])
    return response.text

def extract_contact_info(text):
    phone_pattern = re.compile(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
    phone_match = phone_pattern.search(text)
    return phone_match.group(0) if phone_match else "N/A"

def input_file_setup(uploaded_file):
    if uploaded_file is not None:
        file_type = uploaded_file.type
        if file_type == "application/pdf":
            try:
                document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                text_parts = [page.get_text() for page in document]
                file_content = " ".join(text_parts)
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
                file_content = ""
        elif file_type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            try:
                doc = docx.Document(uploaded_file)
                file_content = "\n".join([para.text for para in doc.paragraphs])
            except Exception as e:
                st.error(f"Error processing Word document: {e}")
                file_content = ""
        else:
            st.error("Unsupported file type")
            file_content = ""
        return file_content
    else:
        return ""

def extract_skills(text, skill_list):
    if not text or not skill_list:
        return "N/A"
    
    # Create a case-insensitive pattern for each skill with word boundaries
    found_skills = []
    for skill in skill_list:
        # Clean the skill and create proper regex pattern
        clean_skill = re.escape(skill.strip())
        pattern = re.compile(r'\b' + clean_skill + r'\b', re.IGNORECASE)
        if pattern.search(text):
            found_skills.append(skill.strip())
    
    return ", ".join(found_skills) if found_skills else "N/A"

st.header("Multi Resume Matcher with JD and skills")
st.subheader("Upload Job Description and Resumes to Analyze Matching Scores")

uploaded_jd = st.file_uploader("Upload Job Description (PDF, DOC, DOCX)...", type=["pdf", "doc", "docx"])

jd_content = ""
if uploaded_jd is not None:
    jd_content = input_file_setup(uploaded_jd)
    st.write("Job Description Uploaded Successfully")

uploaded_resumes = st.file_uploader("Upload Resumes (Multiple PDFs, DOC, DOCX)...", type=["pdf", "doc", "docx"], accept_multiple_files=True)

skills_required = st.text_input("Enter key skills required for the job (comma-separated):")
skills_list = [skill.strip() for skill in skills_required.split(",") if skill.strip()]

submit = st.button("Analyze Resumes")

table_data = []

if submit:
    if uploaded_jd is None:
        st.write("Please upload a Job Description to proceed.")
    elif not uploaded_resumes:
        st.write("Please upload at least one Resume to proceed.")
    elif not skills_list:
        st.write("Please enter key skills required for the job.")
    else:
        # For User-Entered Skills, use the actual input skills list
        user_entered_skills = ", ".join(skills_list) if skills_list else "N/A"
        
        # Extract skills from JD for reference (optional, not used in table)
        jd_skills = extract_skills(jd_content, skills_list)
        
        for resume in uploaded_resumes:
            resume_content = input_file_setup(resume)
            contact_info = extract_contact_info(resume_content)
            resume_skills = extract_skills(resume_content, skills_list)
            
            input_prompt = f"""
            Role: Resume Analyzer
            
            Task: Analyze the compatibility between the resume and job requirements below. Format your response precisely as specified.
            
            Instructions:
            1. Extract the candidate's name from the resume
            2. Calculate a match percentage based on skills overlap and relevance
            3. Structure your analysis in the exact format below
            
            Required Skills: {skills_list}
            
            Output Format (maintain this exact structure):
            Name: [Full name extracted from resume]
            Match Percentage: [0-100%]
            JD Skills: [Comma-separated list of skills found in the job description]
            Resume Skills: [Comma-separated list of skills found in the resume]
            Contact Number: {contact_info}
            
            Importance:
            - Be precise in your percentage calculation
            - Include ALL matching skills, even partial matches
            - Return ONLY the requested information in the specified format
            - Do not include explanations or additional text
            """
            
            response = get_gemini_response(input_prompt, resume_content, jd_content)
            
            name = resume.name  # Default to file name
            match_percentage = "N/A"
            
            if response:
                lines = response.split("\n")
                for line in lines:
                    line_lower = line.lower()
                    if "match percentage" in line_lower:
                        match_percentage = line.split(":")[-1].strip()
                    elif "name:" in line_lower and line_lower.index("name:") == 0:
                        extracted_name = line.split(":", 1)[-1].strip()
                        if extracted_name and extracted_name != "[Full name extracted from resume]":
                            name = extracted_name  # Use extracted name if available
            
            table_data.append([name, match_percentage, user_entered_skills, resume_skills, contact_info])
        
        df = pd.DataFrame(table_data, columns=["Name", "Match Percentage", "User-Entered Skills", "Skills as per Resume", "Contact Number"])
        st.subheader("Resume Analysis Results")
        st.dataframe(df)
