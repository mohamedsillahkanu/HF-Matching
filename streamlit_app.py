import streamlit as st
import pandas as pd
import numpy as np
from jellyfish import jaro_winkler_similarity
from io import BytesIO
from PIL import Image
import time
import random
import threading

# Set page config first
st.set_page_config(page_title="Health Facility Matching Tool", page_icon="🏥", layout="wide")

# Force dark theme across the entire app
st.markdown("""
    <style>
        /* Override Streamlit's default theme to force dark mode */
        .stApp {
            background-color: #0E1117 !important;
        }
        
        /* Dark theme for sidebar */
        [data-testid="stSidebar"] {
            background-color: #1E1E1E !important;
            border-right: 1px solid #2E2E2E;
        }
        
        /* Dark theme for all text */
        .stMarkdown, p, h1, h2, h3 {
            color: #E0E0E0 !important;
        }
        
        /* Custom title styles */
        .custom-title {
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            padding: 1rem 0;
            margin-bottom: 2rem;
            color: #E0E0E0 !important;
            background: linear-gradient(135deg, #3498db, #2ecc71);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: block;
            width: 100%;
        }
        
        /* Dark theme for selectbox */
        .stSelectbox > div > div {
            background-color: #1E1E1E !important;
            color: #E0E0E0 !important;
        }
        
        /* Dark theme for checkbox */
        .stCheckbox > div > div > label {
            color: #E0E0E0 !important;
        }
        
        /* Update section cards for dark theme */
        .section-card {
            background: #1E1E1E !important;
            color: #E0E0E0 !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3) !important;
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
            border-left: 5px solid #3498db;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            animation: slideIn 0.5s ease-out;
        }
        
        .section-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.5);
            background: #2E2E2E !important;
        }
        
        /* Dark theme for content text */
        .content-text {
            color: #E0E0E0 !important;
        }

        .section-header {
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
            color: #3498db !important;
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        @keyframes slideIn {
            from { transform: translateX(-20px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        @keyframes scaleIn {
            from { transform: scale(0.95); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }

        /* Custom styling for the dataframe */
        .dataframe {
            background-color: #1E1E1E !important;
            color: #E0E0E0 !important;
        }
        
        /* Style for file uploader */
        .stFileUploader {
            background-color: #1E1E1E !important;
            border: 1px solid #3498db !important;
            border-radius: 8px;
            padding: 10px;
        }

        /* Style for buttons */
        .stButton button {
            background-color: #3498db !important;
            color: #E0E0E0 !important;
            border: none !important;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            transition: all 0.3s ease;
        }

        .stButton button:hover {
            background-color: #2980b9 !important;
            transform: translateY(-2px);
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session states
if 'last_animation' not in st.session_state:
    st.session_state.last_animation = time.time()
    st.session_state.theme_index = 0
    st.session_state.first_load = True
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'master_hf_list' not in st.session_state:
    st.session_state.master_hf_list = None
if 'health_facilities_dhis2_list' not in st.session_state:
    st.session_state.health_facilities_dhis2_list = None

# Define dark themes
themes = {
    "Dark Modern": {
        "bg": "#0E1117",
        "accent": "#3498db",
        "text": "#E0E0E0",
        "gradient": "linear-gradient(135deg, #3498db, #2ecc71)"
    },
    "Dark Elegance": {
        "bg": "#1a1a1a",
        "accent": "#e74c3c",
        "text": "#E0E0E0",
        "gradient": "linear-gradient(135deg, #e74c3c, #c0392b)"
    },
    "Dark Nature": {
        "bg": "#1E1E1E",
        "accent": "#27ae60",
        "text": "#E0E0E0",
        "gradient": "linear-gradient(135deg, #27ae60, #2ecc71)"
    },
    "Dark Cosmic": {
        "bg": "#2c0337",
        "accent": "#9b59b6",
        "text": "#E0E0E0",
        "gradient": "linear-gradient(135deg, #9b59b6, #8e44ad)"
    },
    "Dark Ocean": {
        "bg": "#1A2632",
        "accent": "#00a8cc",
        "text": "#E0E0E0",
        "gradient": "linear-gradient(135deg, #00a8cc, #0089a7)"
    }
}

# Welcome animation on first load
if st.session_state.first_load:
    st.balloons()
    st.snow()
    welcome_placeholder = st.empty()
    welcome_placeholder.success("Welcome to the Health Facility Matching Tool! 🏥")
    time.sleep(3)
    welcome_placeholder.empty()
    st.session_state.first_load = False

# Auto theme changer and animation
current_time = time.time()
if current_time - st.session_state.last_animation >= 30:
    st.session_state.last_animation = current_time
    theme_keys = list(themes.keys())
    st.session_state.theme_index = (st.session_state.theme_index + 1) % len(theme_keys)
    selected_theme = theme_keys[st.session_state.theme_index]
    st.balloons()
else:
    selected_theme = list(themes.keys())[st.session_state.theme_index]

# Get current theme
theme = themes[selected_theme]

def calculate_match(df1, df2, col1, col2, threshold):
    """Calculate matching scores between two columns using Jaro-Winkler similarity."""
    results = []
    
    for idx1, row1 in df1.iterrows():
        value1 = str(row1[col1])
        if value1 in df2[col2].values:
            # Exact match
            matched_row = df2[df2[col2] == value1].iloc[0]
            result_row = {
                f'MFL_{col1}': value1,
                f'DHIS2_{col2}': value1,
                'Match_Score': 100,
                'Match_Status': 'Match'
            }
            # Add all columns from both dataframes
            for c in df1.columns:
                if c != col1:
                    result_row[f'MFL_{c}'] = row1[c]
            for c in df2.columns:
                if c != col2:
                    result_row[f'DHIS2_{c}'] = matched_row[c]
            results.append(result_row)
        else:
            # Find best match
            best_score = 0
            best_match_row = None
            for idx2, row2 in df2.iterrows():
                value2 = str(row2[col2])
                similarity = jaro_winkler_similarity(value1, value2) * 100
                if similarity > best_score:
                    best_score = similarity
                    best_match_row = row2
            
            result_row = {
                f'MFL_{col1}': value1,
                f'DHIS2_{col2}': best_match_row[col2] if best_match_row is not None else None,
                'Match_Score': round(best_score, 2),
                'Match_Status': 'Unmatch' if best_score < threshold else 'Match'
            }
            # Add all columns from both dataframes
            for c in df1.columns:
                if c != col1:
                    result_row[f'MFL_{c}'] = row1[c]
            for c in df2.columns:
                if c != col2:
                    result_row[f'DHIS2_{c}'] = best_match_row[c] if best_match_row is not None else None
            results.append(result_row)
    
    return pd.DataFrame(results)

def main():
    st.title('Health Facility Name Matching')

    # Step 1: File Upload
    if st.session_state.step == 1:
        st.markdown('<div class="section-card"><div class="section-header">Step 1: Upload Files</div>', unsafe_allow_html=True)
        mfl_file = st.file_uploader("Upload Master HF List (CSV, Excel):", type=['csv', 'xlsx', 'xls'])
        dhis2_file = st.file_uploader("Upload DHIS2 HF List (CSV, Excel):", type=['csv', 'xlsx', 'xls'])

        if mfl_file and dhis2_file:
            try:
                # Read files
                if mfl_file.name.endswith('.csv'):
                    st.session_state.master_hf_list = pd.read_csv(mfl_file)
                else:
                    st.session_state.master_hf_list = pd.read_excel(mfl_file)

                if dhis2_file.name.endswith('.csv'):
                    st.session_state.health_facilities_dhis2_list = pd.read_csv(dhis2_file)
                else:
                    st.session_state.health_facilities_dhis2_list = pd.read_excel(dhis2_file)

                st.success("Files uploaded successfully!")
                
                # Display previews
                st.markdown('<div class="section-header">Preview of Master HF List</div>', unsafe_allow_html=True)
                st.dataframe(st.session_state.master_hf_list.head())
                st.markdown('<div class="section-header">Preview of DHIS2 HF List</div>', unsafe_allow_html=True)
                st.dataframe(st.session_state.health_facilities_dhis2_list.head())

                if st.button("Proceed to Column Renaming"):
                    st.session_state.step = 2
                    st.experimental_rerun()

            except Exception as e:
                st.error(f"Error reading files: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Step 2: Column Renaming
    elif st.session_state.step == 2:
        st.markdown('<div class="section-card"><div class="section-header">Step 2: Rename Columns (Optional)</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-header">Master HF List Columns</div>', unsafe_allow_html=True)
            mfl_renamed_columns = {}
            for col in st.session_state.master_hf_list.columns:
                new_col = st.text_input(f"Rename '{col}' to:", key=f"mfl_{col}", value=col)
                mfl_renamed_columns[col] = new_col

        with col2:
            st.markdown('<div class="section-header">DHIS2 HF List Columns</div>', unsafe_allow_html=True)
            dhis2_renamed_columns = {}
            for col in st.session_state.health_facilities_dhis2_list.columns:
                new_col = st.text_input(f"Rename '{col}' to:", key=f"dhis2_{col}", value=col)
                dhis2_renamed_columns[col] = new_col

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Apply Changes and Continue"):
                st.session_state.master_hf_list = st.session_state.master_hf_list.rename(columns=mfl_renamed_columns)
                st.session_state.health_facilities_dhis2_list = st.session_state.health_facilities_dhis2_list.rename(
                    columns=dhis2_renamed_columns)
                st.session_state.step = 3
                st.experimental_rerun()
        
        with col2:
            if st.button("Skip Renaming"):
                st.session_state.step = 3
                st.experimental_rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Step 3: Column Selection and Matching
    elif st.session_state.step == 3:
        st.markdown('<div class="section-card"><div class="section-header">Step 3: Select Columns for Matching</div>', unsafe_allow_html=True)
        
        mfl_col = st.selectbox("Select HF Name column in Master HF List:", 
                              st.session_state.master_hf_list.columns)
        dhis2_col = st.selectbox("Select HF Name column in DHIS2 HF List:", 
                                st.session_state.health_facilities_dhis2_list.columns)
        
        threshold = st.slider("Set Match Threshold (0-100):", 
                            min_value=0, max_value=100, value=70)

        if st.button("Perform Matching"):
            # Process data
            master_hf_list_clean = st.session_state.master_hf_list.copy()
            dhis2_list_clean = st.session_state.health_facilities_dhis2_list.copy()
            
            master_hf_list_clean[mfl_col] = master_hf_list_clean[mfl_col].astype(str)
            master_hf_list_clean = master_hf_list_clean.drop_duplicates(subset=[mfl_col])
            dhis2_list_clean[dhis2_col] = dhis2_list_clean[dhis2_col].astype(str)

            st.markdown('<div class="section-header">Counts of Health Facilities</div>', unsafe_allow_html=True)
            st.write(f"Count of HFs in DHIS2 list: {len(dhis2_list_clean)}")
            st.write(f"Count of HFs in MFL list: {len(master_hf_list_clean)}")

            # Perform matching
            with st.spinner("Performing matching..."):
                hf_name_match_results = calculate_match(
                    master_hf_list_clean,
                    dhis2_list_clean,
                    mfl_col,
                    dhis2_col,
                    threshold
                )

                # Add a column for suggested names
                hf_name_match_results['Suggested_HF_Name'] = np.where(
                    hf_name_match_results['Match_Score'] >= threshold,
                    hf_name_match_results[f'DHIS2_{dhis2_col}'],
                    hf_name_match_results[f'MFL_{mfl_col}']
                )

                # Calculate summary statistics
                total_facilities = len(hf_name_match_results)
                matched_facilities = len(hf_name_match_results[hf_name_match_results['Match_Status'] == 'Match'])
                unmatched_facilities = total_facilities - matched_facilities
                match_rate = (matched_facilities/total_facilities*100) if total_facilities > 0 else 0

                # Display summary statistics
                st.markdown('<div class="section-header">Summary Statistics</div>', unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Facilities", total_facilities)
                with col2:
                    st.metric("Matched Facilities", matched_facilities)
                with col3:
                    st.metric("Unmatched Facilities", unmatched_facilities)
                with col4:
                    st.metric("Match Rate", f"{match_rate:.1f}%")

                # Display results
                st.markdown('<div class="section-header">Matching Results</div>', unsafe_allow_html=True)
                st.dataframe(hf_name_match_results)

                # Download results
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    hf_name_match_results.to_excel(writer, index=False)
                output.seek(0)

                st.download_button(
                    label="Download Matching Results as Excel",
                    data=output,
                    file_name="hf_name_matching_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        if st.button("Start Over"):
            st.session_state.step = 1
            st.session_state.master_hf_list = None
            st.session_state.health_facilities_dhis2_list = None
            st.experimental_rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

# Sidebar theme selector
st.sidebar.selectbox(
    "🎨 Select Theme",
    list(themes.keys()),
    index=st.session_state.theme_index,
    key='theme_selector'
)

# Enable/Disable animations toggle
if st.sidebar.checkbox("Enable Auto Animations", value=True):
    def show_periodic_animations():
        while True:
            time.sleep(60)
            st.balloons()
            time.sleep(10)
            st.snow()

    # Start animation thread if not already running
    if not hasattr(st.session_state, 'animation_thread'):
        st.session_state.animation_thread = threading.Thread(target=show_periodic_animations)
        st.session_state.animation_thread.daemon = True
        st.session_state.animation_thread.start()

if __name__ == "__main__":
    main()
