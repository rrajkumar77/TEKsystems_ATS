import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
from collections import defaultdict

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
def get_project_summary(consolidated_text):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = """
        Analyze the following consolidated project information and create a summary in this exact format:

        * **Project Name:** [Project Name]
        * **Goals:** [Main objectives and goals - be specific and concise]
        * **Tech Skills:** [Technical skills, tools, technologies used - extract from descriptions]
        * **Key Achievement:** [Most significant accomplishment or deliverable]
        * **Value Delivered:** [Business value, impact, or benefits provided]

        If multiple entries exist for the same project, consolidate the information intelligently.
        Focus on being concise but comprehensive. Here's the project data:
        """
        response = model.generate_content([prompt + "\n\n" + consolidated_text])
        return response.text
    except Exception as e:
        return f"<p style='color:red;'>Error generating summary: {str(e)}</p>"

# Consolidate project entries
def consolidate_projects(filtered_df):
    project_groups = defaultdict(list)
    
    # Group by project name
    for idx, row in filtered_df.iterrows():
        project_name = row.get("Project_Name", "").strip()
        if project_name:
            project_groups[project_name].append(row)
    
    consolidated_projects = {}
    
    for project_name, entries in project_groups.items():
        # Consolidate all information for this project
        all_descriptions = []
        all_achievements = []
        all_value_adds = []
        all_skills = []
        team_leads = set()
        business_units = set()
        employees = set()
        
        for entry in entries:
            # Collect unique information
            if pd.notna(entry.get("Project_Description")) and entry.get("Project_Description").strip():
                all_descriptions.append(entry.get("Project_Description").strip())
            
            if pd.notna(entry.get("Acheivements_ValueAdds")) and entry.get("Acheivements_ValueAdds").strip():
                all_achievements.append(entry.get("Acheivements_ValueAdds").strip())
            
            if pd.notna(entry.get("Value_Add")) and entry.get("Value_Add").strip():
                all_value_adds.append(entry.get("Value_Add").strip())
            
            if pd.notna(entry.get("Project_Brief_Skill")) and entry.get("Project_Brief_Skill").strip():
                all_skills.append(entry.get("Project_Brief_Skill").strip())
            
            if pd.notna(entry.get("Team_Lead")) and entry.get("Team_Lead").strip():
                team_leads.add(entry.get("Team_Lead").strip())
            
            if pd.notna(entry.get("Business_Unit_Name")) and entry.get("Business_Unit_Name").strip():
                business_units.add(entry.get("Business_Unit_Name").strip())
            
            if pd.notna(entry.get("Created By")) and entry.get("Created By").strip():
                employees.add(entry.get("Created By").strip())
        
        # Create consolidated text
        consolidated_text = f"""
        Project Name: {project_name}
        Business Units: {', '.join(business_units)}
        Team Leads: {', '.join(team_leads)}
        Employees: {', '.join(employees)}
        
        Project Descriptions:
        {' | '.join(set(all_descriptions))}
        
        Technical Skills:
        {' | '.join(set(all_skills))}
        
        Achievements:
        {' | '.join(set(all_achievements))}
        
        Value Adds:
        {' | '.join(set(all_value_adds))}
        """
        
        consolidated_projects[project_name] = consolidated_text
    
    return consolidated_projects

# Streamlit UI
st.set_page_config(page_title="QBR Summary Generator", layout="wide")
st.title("📊 Enhanced Project Summary for Sales")

uploaded_file = st.file_uploader("Upload the QBR CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    
    # Display basic info
    st.info(f"📋 Total records: {len(df)} | Columns: {len(df.columns)}")
    
    # Create filtering options
    st.sidebar.header("🔍 Filter Options")
    
    # Business Unit Filter
    business_units = sorted(df['Business_Unit_Name'].dropna().unique())
    selected_business_units = st.sidebar.multiselect(
        "Select Business Unit(s):",
        options=business_units,
        default=business_units,  # Select all by default
        help="Choose one or more business units"
    )
    
    # Project Name Filter
    project_names = sorted(df['Project_Name'].dropna().unique())
    
    # Option to select all projects or specific ones
    select_all_projects = st.sidebar.checkbox("Select All Projects", value=True)
    
    if select_all_projects:
        selected_projects = project_names
        st.sidebar.info(f"All {len(project_names)} projects selected")
    else:
        selected_projects = st.sidebar.multiselect(
            "Select Project(s):",
            options=project_names,
            default=[],
            help="Choose specific projects"
        )
    
    # Apply filters
    if selected_business_units and selected_projects:
        filtered_df = df[
            (df['Business_Unit_Name'].isin(selected_business_units)) &
            (df['Project_Name'].isin(selected_projects))
        ]
        
        st.subheader("📊 Filtered Results")
        st.info(f"Showing {len(filtered_df)} records after filtering")
        
        if len(filtered_df) > 0:
            # Show filtering summary
            with st.expander("📋 Filter Summary", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Selected Business Units:**")
                    for bu in selected_business_units:
                        st.write(f"• {bu}")
                
                with col2:
                    st.write("**Selected Projects:**")
                    unique_projects = filtered_df['Project_Name'].unique()
                    for proj in sorted(unique_projects)[:10]:  # Show first 10
                        st.write(f"• {proj}")
                    if len(unique_projects) > 10:
                        st.write(f"• ... and {len(unique_projects) - 10} more")
            
            # Generate summaries button
            if st.button("🚀 Generate Project Summaries", type="primary"):
                consolidated_projects = consolidate_projects(filtered_df)
                
                st.subheader("📝 Generated Summaries")
                
                # Progress bar
                progress_bar = st.progress(0)
                total_projects = len(consolidated_projects)
                
                for i, (project_name, consolidated_text) in enumerate(consolidated_projects.items()):
                    # Update progress
                    progress_bar.progress((i + 1) / total_projects)
                    
                    # Generate summary
                    with st.spinner(f"Generating summary for: {project_name}"):
                        summary = get_project_summary(consolidated_text)
                        formatted_output = format_summary(project_name, summary)
                        st.markdown(formatted_output, unsafe_allow_html=True)
                
                progress_bar.empty()
                st.success(f"✅ Generated summaries for {total_projects} unique projects!")
                
        else:
            st.warning("⚠️ No records match the selected filters. Please adjust your selection.")
    
    else:
        st.warning("⚠️ Please select at least one Business Unit and Project to continue.")
    
    # Show data preview
    with st.expander("👀 Data Preview", expanded=False):
        st.dataframe(df.head(10))
        
        # Show column info
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Non-null Count': df.count(),
            'Data Type': df.dtypes
        })
        st.dataframe(col_info)

# Add footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 14px;'>"
    "Enhanced QBR Summary Generator with Filtering & Consolidation"
    "</div>", 
    unsafe_allow_html=True
)
