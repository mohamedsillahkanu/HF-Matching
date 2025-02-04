import streamlit as st
import pandas as pd
import numpy as np
from jellyfish import jaro_winkler_similarity
from io import BytesIO
from PIL import Image
import random
import time
import threading

# Page configuration
st.set_page_config(page_title="Geospatial Analysis Tool", page_icon="🗺️", layout="wide")

# Theme definitions
themes = {
    "Black Modern": {
        "bg": "#000000",
        "accent": "#3498db",
        "text": "#FFFFFF",
        "gradient": "linear-gradient(135deg, #3498db, #2ecc71)"
    },
    "Light Silver": {
        "bg": "#F5F5F5",
        "accent": "#1E88E5",
        "text": "#212121",
        "gradient": "linear-gradient(135deg, #1E88E5, #64B5F6)"
    },
    "Light Sand": {
        "bg": "#FAFAFA",
        "accent": "#FF7043",
        "text": "#424242",
        "gradient": "linear-gradient(135deg, #FF7043, #FFB74D)"
    },
    "Light Modern": {
        "bg": "#FFFFFF",
        "accent": "#3498db",
        "text": "#333333",
        "gradient": "linear-gradient(135deg, #3498db, #2ecc71)"
    },
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

# Custom CSS styles with enhanced button and input styling
st.markdown("""
    <style>
    .stApp {
        background-color: var(--bg-color, #0E1117) !important;
        color: var(--text-color, #E0E0E0) !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg, #1E1E1E) !important;
        border-right: 1px solid var(--border-color, #2E2E2E);
    }
    
    .stMarkdown, p, h1, h2, h3 {
        color: var(--text-color, #E0E0E0) !important;
    }

    /* Dark Theme Button Styling */
    [data-theme="dark"] .stButton > button,
    [data-theme="dark"] .stDownloadButton > button {
        color: white !important;
        font-weight: bold !important;
        background-color: black !important;
        border: 2px solid #3498db !important;
        border-radius: 4px !important;
        padding: 0.5rem 1rem !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }

    [data-theme="dark"] .stButton > button:hover,
    [data-theme="dark"] .stDownloadButton > button:hover {
        background-color: #2E2E2E !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
        transform: translateY(-1px) !important;
    }

    /* Light Theme Button Styling */
    [data-theme="light"] .stButton > button,
    [data-theme="light"] .stDownloadButton > button {
        color: black !important;
        font-weight: bold !important;
        background-color: white !important;
        border: 2px solid #2e2e2e !important;
        border-radius: 4px !important;
        padding: 0.5rem 1rem !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }

    [data-theme="light"] .stButton > button:hover,
    [data-theme="light"] .stDownloadButton > button:hover {
        background-color: #f0f0f0 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
        transform: translateY(-1px) !important;
    }

    /* Dark Theme File Uploader */
    [data-theme="dark"] [data-testid="stFileUploader"] {
        background-color: #1E1E1E !important;
        padding: 1rem !important;
        border-radius: 4px !important;
        border: 2px solid #3498db !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }

    [data-theme="dark"] [data-testid="stFileUploader"] > div > div {
        color: white !important;
        font-weight: bold !important;
    }

    /* Light Theme File Uploader */
    [data-theme="light"] [data-testid="stFileUploader"] {
        background-color: white !important;
        padding: 1rem !important;
        border-radius: 4px !important;
        border: 2px solid #2e2e2e !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }

    [data-theme="light"] [data-testid="stFileUploader"] > div > div {
        color: black !important;
        font-weight: bold !important;
    }

    /* Dark Theme Select Box */
    [data-theme="dark"] .stSelectbox > div > div {
        color: white !important;
        font-weight: bold !important;
        background-color: #1E1E1E !important;
        border: 2px solid #3498db !important;
        border-radius: 4px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }

    /* Light Theme Select Box */
    [data-theme="light"] .stSelectbox > div > div {
        color: black !important;
        font-weight: bold !important;
        background-color: white !important;
        border: 2px solid #2e2e2e !important;
        border-radius: 4px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }

    /* Success Message Styling */
    [data-theme="dark"] .stSuccess {
        color: white !important;
        font-weight: bold !important;
        background-color: #1E1E1E !important;
        border: 2px solid #00ff00 !important;
        border-radius: 4px !important;
    }

    [data-theme="light"] .stSuccess {
        color: black !important;
        font-weight: bold !important;
        background-color: white !important;
        border: 2px solid #00aa00 !important;
        border-radius: 4px !important;
    }

    /* Error Message Styling */
    [data-theme="dark"] .stError {
        color: white !important;
        font-weight: bold !important;
        background-color: #1E1E1E !important;
        border: 2px solid #ff0000 !important;
        border-radius: 4px !important;
    }

    [data-theme="light"] .stError {
        color: black !important;
        font-weight: bold !important;
        background-color: white !important;
        border: 2px solid #ff0000 !important;
        border-radius: 4px !important;
    }

    /* DataFrame Styling */
    [data-theme="dark"] .stDataFrame {
        color: white !important;
        font-weight: bold !important;
        background-color: #1E1E1E !important;
        border: 2px solid #3498db !important;
        border-radius: 4px !important;
    }

    [data-theme="light"] .stDataFrame {
        color: black !important;
        font-weight: bold !important;
        background-color: white !important;
        border: 2px solid #2e2e2e !important;
        border-radius: 4px !important;
    }

    /* Slider Styling */
    [data-theme="dark"] .stSlider > div > div > div {
        background-color: #3498db !important;
    }

    [data-theme="dark"] .stSlider > div > div > div > div {
        background-color: white !important;
    }

    [data-theme="light"] .stSlider > div > div > div {
        background-color: #2e2e2e !important;
    }

    [data-theme="light"] .stSlider > div > div > div > div {
        background-color: black !important;
    }

    /* Text Input Styling */
    [data-theme="dark"] .stTextInput > div > div > input {
        color: white !important;
        font-weight: bold !important;
        background-color: #1E1E1E !important;
        border: 2px solid #3498db !important;
        border-radius: 4px !important;
    }

    [data-theme="light"] .stTextInput > div > div > input {
        color: black !important;
        font-weight: bold !important;
        background-color: white !important;
        border: 2px solid #2e2e2e !important;
        border-radius: 4px !important;
    }

    /* Custom Cards */
    .section-card {
        background: var(--card-bg, #1E1E1E) !important;
        color: var(--text-color, #E0E0E0) !important;
        box-shadow: 0 4px 6px var(--shadow-color, rgba(0, 0, 0, 0.3)) !important;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        border-left: 5px solid var(--accent-color, #3498db);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: slideIn 0.5s ease-out;
    }

    .section-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px var(--shadow-color, rgba(0, 0, 0, 0.5));
    }

    /* Custom Title */
    .custom-title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
        background: var(--gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
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
</style>
""", unsafe_allow_html=True)



def calculate_match(column1, column2, threshold):
    """Calculate matching scores between two columns using Jaro-Winkler similarity."""
    results = []
    
    for value1 in column1:
        if value1 in column2.values:
            results.append({
                'Col1': value1,
                'Col2': value1,
                'Match_Score': 100,
                'Match_Status': 'Match'
            })
        else:
            best_score = 0
            best_match = None
            for value2 in column2:
                similarity = jaro_winkler_similarity(str(value1), str(value2)) * 100
                if similarity > best_score:
                    best_score = similarity
                    best_match = value2
            results.append({
                'Col1': value1,
                'Col2': best_match,
                'Match_Score': round(best_score, 2),
                'Match_Status': 'Unmatch' if best_score < threshold else 'Match'
            })
    
    for value2 in column2:
        if value2 not in [r['Col2'] for r in results]:
            results.append({
                'Col1': None,
                'Col2': value2,
                'Match_Score': 0,
                'Match_Status': 'Unmatch'
            })
    
    return pd.DataFrame(results)

def show_confetti():
    """Display confetti animation."""
    st.markdown("""
        <style>
            @keyframes confetti {
                0% { transform: translateY(0) rotate(0deg); }
                100% { transform: translateY(100vh) rotate(360deg); }
            }
            .confetti {
                position: fixed;
                animation: confetti 4s linear;
                z-index: 9999;
            }
        </style>
    """, unsafe_allow_html=True)
    for i in range(50):
        color = f"hsl({random.randint(0, 360)}, 100%, 50%)"
        left = random.randint(0, 100)
        st.markdown(f"""
            <div class="confetti" style="left: {left}vw; background: {color}; 
            width: 10px; height: 10px; border-radius: 50%;"></div>
        """, unsafe_allow_html=True)

def show_sparkles():
    """Display sparkles animation."""
    st.markdown("""
        <style>
            @keyframes sparkle {
                0% { transform: scale(0); opacity: 0; }
                50% { transform: scale(1); opacity: 1; }
                100% { transform: scale(0); opacity: 0; }
            }
            .sparkle {
                position: fixed;
                animation: sparkle 2s infinite;
            }
        </style>
    """, unsafe_allow_html=True)
    for i in range(20):
        left = random.randint(0, 100)
        top = random.randint(0, 100)
        st.markdown(f"""
            <div class="sparkle" style="left: {left}vw; top: {top}vh; 
            background: gold; width: 5px; height: 5px; border-radius: 50%;"></div>
        """, unsafe_allow_html=True)

def show_fireworks():
    """Display random fireworks animation."""
    animations = [st.balloons(), st.snow(), show_confetti(), show_sparkles()]
    random.choice(animations)

# List of available animations
animations_list = [
    st.balloons,
    st.snow,
    show_confetti,
    show_sparkles,
    show_fireworks,
    lambda: [st.balloons(), st.snow()],
    lambda: [show_confetti(), show_sparkles()],
    lambda: [st.balloons(), show_confetti()],
    lambda: [st.snow(), show_sparkles()],
    lambda: [show_confetti(), st.snow()]
]

# Initialize session state for animations
if 'last_animation' not in st.session_state:
    st.session_state.last_animation = time.time()
    st.session_state.theme_index = list(themes.keys()).index("Black Modern")
    st.session_state.first_load = True

# Show welcome message on first load
if st.session_state.first_load:
    st.balloons()
    st.snow()
    welcome_placeholder = st.empty()
    welcome_placeholder.success("Welcome to the Geospatial Analysis Tool! 🌍")
    time.sleep(3)
    welcome_placeholder.empty()
    st.session_state.first_load = False

# Handle theme rotation
current_time = time.time()
if current_time - st.session_state.last_animation >= 30:
    st.session_state.last_animation = current_time
    theme_keys = list(themes.keys())
    st.session_state.theme_index = (st.session_state.theme_index + 1) % len(theme_keys)
    st.balloons()

# Theme selection
selected_theme = st.sidebar.selectbox(
    "🎨 Select Theme",
    list(themes.keys()),
    index=st.session_state.theme_index,
    key='theme_selector'
)

# Handle theme change animations
if 'previous_theme' not in st.session_state:
    st.session_state.previous_theme = selected_theme
if st.session_state.previous_theme != selected_theme:
    st.balloons()
    st.snow()
    st.session_state.previous_theme = selected_theme

# Apply selected theme
theme = themes[selected_theme]
is_light_theme = "Light" in selected_theme

st.markdown(f"""
    <style>
        :root {{
            --bg-color: {theme['bg']};
            --text-color: {theme['text']};
            --accent-color: {theme['accent']};
            --gradient: {theme['gradient']};
            --sidebar-bg: {theme['bg']};
            --card-bg: {'#F8F9FA' if is_light_theme else '#1E1E1E'};
            --card-hover-bg: {'#E9ECEF' if is_light_theme else '#2E2E2E'};
            --input-bg: {'#F8F9FA' if is_light_theme else '#1E1E1E'};
            --shadow-color: {f'rgba(0, 0, 0, 0.1)' if is_light_theme else 'rgba(0, 0, 0, 0.3)'};
            --border-color: {'#DEE2E6' if is_light_theme else '#2E2E2E'};
        }}
    </style>
""", unsafe_allow_html=True)

def main():


    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'master_hf_list' not in st.session_state:
        st.session_state.master_hf_list = None
    if 'health_facilities_dhis2_list' not in st.session_state:
        st.session_state.health_facilities_dhis2_list = None

    # Step 1: File Upload
    if st.session_state.step == 1:
        st.header("Step 1: Upload Files")
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
                
                st.subheader("Preview of Master HF List")
                st.dataframe(st.session_state.master_hf_list.head())
                st.subheader("Preview of DHIS2 HF List")
                st.dataframe(st.session_state.health_facilities_dhis2_list.head())

                if st.button("Proceed to Column Renaming"):
                    st.session_state.step = 2
                    st.experimental_rerun()

            except Exception as e:
                st.error(f"Error reading files: {e}")

    # Step 2: Column Renaming
    elif st.session_state.step == 2:
        st.header("Step 2: Rename Columns (Optional)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Master HF List Columns")
            mfl_renamed_columns = {}
            for col in st.session_state.master_hf_list.columns:
                new_col = st.text_input(f"Rename '{col}' to:", key=f"mfl_{col}", value=col)
                mfl_renamed_columns[col] = new_col

        with col2:
            st.subheader("DHIS2 HF List Columns")
            dhis2_renamed_columns = {}
            for col in st.session_state.health_facilities_dhis2_list.columns:
                new_col = st.text_input(f"Rename '{col}' to:", key=f"dhis2_{col}", value=col)
                dhis2_renamed_columns[col] = new_col

        if st.button("Apply Changes and Continue"):
            st.session_state.master_hf_list = st.session_state.master_hf_list.rename(columns=mfl_renamed_columns)
            st.session_state.health_facilities_dhis2_list = st.session_state.health_facilities_dhis2_list.rename(
                columns=dhis2_renamed_columns)
            st.session_state.step = 3
            st.experimental_rerun()

        if st.button("Skip Renaming"):
            st.session_state.step = 3
            st.experimental_rerun()

    # Step 3: Column Selection and Matching
    elif st.session_state.step == 3:
        st.header("Step 3: Select Columns for Matching")
        
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
            
            # Convert name columns to string and handle duplicates
            master_hf_list_clean[mfl_col] = master_hf_list_clean[mfl_col].astype(str)
            master_hf_list_clean = master_hf_list_clean.drop_duplicates(subset=[mfl_col])
            dhis2_list_clean[dhis2_col] = dhis2_list_clean[dhis2_col].astype(str)
            
            # Display facility counts
            st.write("### Counts of Health Facilities")
            st.write(f"Count of HFs in DHIS2 list: {len(dhis2_list_clean)}")
            st.write(f"Count of HFs in MFL list: {len(master_hf_list_clean)}")
            
            # Perform matching
            with st.spinner("Performing matching..."):
                # Get initial matching results
                hf_name_match_results = calculate_match(
                    master_hf_list_clean[mfl_col],
                    dhis2_list_clean[dhis2_col],
                    threshold
                )
                
                # Rename the matching columns for clarity
                hf_name_match_results = hf_name_match_results.rename(
                    columns={
                        'Col1': 'HF_Name_in_MFL',
                        'Col2': 'HF_Name_in_DHIS2'
                    }
                )
                
                # Add the replacement column based on threshold
                hf_name_match_results['New_HF_Name_in_MFL'] = np.where(
                    hf_name_match_results['Match_Score'] >= threshold,
                    hf_name_match_results['HF_Name_in_DHIS2'],
                    hf_name_match_results['HF_Name_in_MFL']
                )
                
                # Create separate dataframes with suffix for each source
                master_hf_cols = {col: f"{col}_MFL" for col in master_hf_list_clean.columns if col != mfl_col}
                dhis2_cols = {col: f"{col}_DHIS2" for col in dhis2_list_clean.columns if col != dhis2_col}
                
                # Rename columns in original dataframes
                master_hf_list_clean = master_hf_list_clean.rename(columns=master_hf_cols)
                dhis2_list_clean = dhis2_list_clean.rename(columns=dhis2_cols)
                
                # Merge matching results with original dataframes
                merged_results = (
                    hf_name_match_results
                    .merge(
                        master_hf_list_clean,
                        left_on='HF_Name_in_MFL',
                        right_on=mfl_col,
                        how='left'
                    )
                    .merge(
                        dhis2_list_clean,
                        left_on='HF_Name_in_DHIS2',
                        right_on=dhis2_col,
                        how='left'
                    )
                )
                
                # Drop duplicate columns from the merge
                if mfl_col in merged_results.columns:
                    merged_results = merged_results.drop(columns=[mfl_col])
                if dhis2_col in merged_results.columns:
                    merged_results = merged_results.drop(columns=[dhis2_col])
                
                # Reorder columns to put matching-related columns first
                matching_cols = [
                    'HF_Name_in_MFL',
                    'HF_Name_in_DHIS2',
                    'New_HF_Name_in_MFL',
                    'Match_Score',
                    'Match_Status'
                ]
                other_cols = [col for col in merged_results.columns if col not in matching_cols]
                final_col_order = matching_cols + other_cols
                merged_results = merged_results[final_col_order]
                
                # Display results
                st.write("### Matching Results")
                st.write("The results include all columns from both datasets with suffixes:")
                st.write("- '_MFL' for columns from the Master Facility List")
                st.write("- '_DHIS2' for columns from the DHIS2 list")
                st.dataframe(merged_results)
                
                # Add download button for the results
                csv = merged_results.to_csv(index=False)
                st.download_button(
                    label="Download Matching Results",
                    data=csv,
                    file_name="facility_matching_results.csv",
                    mime="text/csv"
                )

        if st.button("Start Over"):
            st.session_state.step = 1
            st.session_state.master_hf_list = None
            st.session_state.health_facilities_dhis2_list = None
            st.experimental_rerun()

# Main execution
if __name__ == "__main__":
    st.title("Health Facility Name Matching")
    
    st.markdown("""
        <div class="img-container" style="text-align: center;">
            <img src="https://github.com/mohamedsillahkanu/si/raw/b0706926bf09ba23d8e90c394fdbb17e864121d8/Sierra%20Leone%20Map.png" 
                 style="width: 50%; max-width: 500px; margin: 20px auto;">
        </div>
    """, unsafe_allow_html=True)
    
    main()
    
    # Enable animations if checkbox is checked
    if st.sidebar.checkbox("Enable Auto Animations", value=False):
        def show_periodic_animations():
            while True:
                time.sleep(60)
                random.choice(animations_list)()
                time.sleep(10)
                random.choice(animations_list)()

        if not hasattr(st.session_state, 'animation_thread'):
            st.session_state.animation_thread = threading.Thread(target=show_periodic_animations)
            st.session_state.animation_thread.daemon = True
            st.session_state.animation_thread.start()
