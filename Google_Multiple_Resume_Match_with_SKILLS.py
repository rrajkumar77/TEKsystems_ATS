
import streamlit as st # type: ignore
import google.generativeai as genai # type: ignore
import os
import re
from dotenv import load_dotenv # type: ignore
import fitz # type: ignore
import docx # type: ignore
import pandas as pd # type: ignore


# Set page configuration at the very beginning
st.set_page_config(page_title="Multi Resume Matcher with Skills")

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input_prompt, resume_content):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input_prompt, resume_content])
    return response.text

def extract_contact_info(text):
    phone_pattern = re.compile(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
    phone_match = phone_pattern.search(text)
    return phone_match.group(0) if phone_match else "N/A"

def input_file_setup(uploaded_file):
    if uploaded_file is not None:
        file_type = uploaded_file.type
        if file_type == "application/pdf":
            document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text_parts = [page.get_text() for page in document]
            file_content = " ".join(text_parts)
        elif file_type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            doc = docx.Document(uploaded_file)
            file_content = "\n".join([para.text for para in doc.paragraphs])
        else:
            raise ValueError("Unsupported file type")
        return file_content
    else:
        return ""

def extract_skills(text, skill_list):
    if not text:
        return "N/A"
    skill_pattern = re.compile(r'(?i)\b(?:' + '|'.join(re.escape(skill) for skill in skill_list) + r')\b')
    skills_found = skill_pattern.findall(text)
    return ", ".join(set(map(str.strip, skills_found))) if skills_found else "N/A"

st.header("Multi Resume Matcher with Skill")
st.subheader("Upload Resumes to Analyze Matching Scores")

uploaded_resumes = st.file_uploader("Upload Resumes (Multiple PDFs, DOC, DOCX)...", type=["pdf", "doc", "docx"], accept_multiple_files=True)

skills_required = st.text_input("Enter key skills for comparison (comma-separated):")
skills_list = [skill.strip() for skill in skills_required.split(",") if skill.strip()]

submit = st.button("Analyze Resumes")

table_data = []

if submit:
    if not uploaded_resumes:
        st.write("Please upload at least one Resume to proceed.")
    elif not skills_list:
        st.write("Please enter key skills for comparison.")
    else:
        for resume in uploaded_resumes:
            resume_content = input_file_setup(resume)
            contact_info = extract_contact_info(resume_content)
            resume_skills = extract_skills(resume_content, skills_list)
            
            input_prompt = f"""
            ```python
input_prompt = f"""
Role: Expert Resume Analyzer and Skills Matcher

Context: You are analyzing a resume to determine how well it matches with a specific set of required skills: {skills_list}.

Task: Perform a detailed analysis of the resume content against the required skills.

Instructions:
1. Carefully read through the entire resume text
2. Identify all skills mentioned in the resume, including both explicit mentions and implied skills from experience descriptions
3. Compare the identified skills with the required skills list: {skills_list}
4. Calculate the match percentage based on the number of required skills found in the resume
5. Consider variations and synonyms of the required skills (e.g., "Python programming" matches "Python")
6. Assign higher importance to skills that appear multiple times or in recent/relevant experience

Output Requirements (FOLLOW THIS FORMAT EXACTLY):
Match Percentage: [Calculate as: (Number of matched skills / Total number of required skills) * 100]%

Skills Comparison:
- Matched Skills: [List all required skills found in the resume, including variations]
- Missing Skills: [List all required skills NOT found in the resume]
- Additional Relevant Skills: [List up to 5 valuable skills found in the resume but not in the required list]

Candidate Assessment:
- Strengths: [2-3 key areas where the candidate's experience aligns well with requirements]
- Development Areas: [1-2 key gaps based on missing skills]
- Overall Match Rating: [High/Medium/Low based on match percentage: High ≥ 80%, Medium ≥ 50%, Low < 50%]
"""

            response = get_gemini_response(input_prompt, resume_content)
            
            name = resume.name  # Extract file name as candidate identifier
            match_percentage = "N/A"
            
            if response:
                lines = response.split("\n")
                for line in lines:
                    line_lower = line.lower()
                    if "match percentage" in line_lower:
                        match_percentage = line.split(":")[-1].strip()
            
            table_data.append([name, match_percentage, skills_required, resume_skills, contact_info])
        
        df = pd.DataFrame(table_data, columns=["Name", "Match Percentage", "User-Entered Skills", "Skills as per Resume", "Contact Number"])
        st.subheader("Resume Analysis Results")
        st.dataframe(df)
