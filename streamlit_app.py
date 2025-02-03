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
        /* Make all text bold */
        * {
            font-weight: bold !important;
            font-size: 1.1rem !important;
        }

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
            font-size: 2.7rem !important;
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
            font-size: 1.1rem !important;
        }
        
        /* Dark theme for checkbox */
        .stCheckbox > div > div > label {
            color: #E0E0E0 !important;
            font-size: 1.1rem !important;
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
            font-size: 1.7rem !important;
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
        
        .dataframe td, .dataframe th {
            font-weight: bold !important;
            font-size: 1.1rem !important;
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
            font-size: 1.1rem !important;
            font-weight: bold !important;
        }

        .stButton button:hover {
            background-color: #2980b9 !important;
            transform: translateY(-2px);
        }

        /* Animation container */
        .animation-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            pointer-events: none;
        }
        
        /* Audio player styling */
        .audio-player {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 1000;
        }
        
        .welcome-animation {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 100%;
            height: 100%;
            z-index: 1000;
            background: rgba(14, 17, 23, 0.9);
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }
        
        /* Audio player buttons */
        .audio-player button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 8px 16px;
            margin: 0 5px;
            border-radius: 4px;
            cursor: pointer;
        }
        
        .audio-player button:hover {
            background-color: #2980b9;
        }
        
        /* Text inputs */
        .stTextInput input {
            font-size: 1.1rem !important;
            font-weight: bold !important;
            color: #000000 !important;
            background-color: #FFFFFF !important;
        }

        /* Slider */
        .stSlider {
            font-size: 1.1rem !important;
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

# Welcome animation with music
if st.session_state.first_load:
    # Background music
    st.markdown("""
        <audio id="bgMusic" autoplay loop>
            <source src="https://assets.mixkit.co/music/preview/mixkit-tech-house-vibes-130.mp3" type="audio/mpeg">
            Your browser does not support the audio element.
        </audio>
        
        <div class="welcome-animation">
            <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
            <lottie-player
                src="https://assets2.lottiefiles.com/packages/lf20_jm6wb9bc.json"
                background="transparent"
                speed="1"
                style="width: 400px; height: 400px;"
                loop
                autoplay>
            </lottie-player>
            <h1 style="color: white; margin-top: 20px;">Welcome to Health Facility Matching Tool!</h1>
        </div>
        
        <script>
            setTimeout(function() {
                document.querySelector('.welcome-animation').style.opacity = '0';
                document.querySelector('.welcome-animation').style.transition = 'opacity 1s';
                setTimeout(function() {
                    document.querySelector('.welcome-animation').style.display = 'none';
                }, 1000);
            }, 5000);
        </script>
    """, unsafe_allow_html=True)
    
    # Add audio controls in sidebar
    st.sidebar.markdown("""
        <div class="audio-player">
            <button onclick="document.getElementById('bgMusic').play()">🎵 Play</button>
            <button onclick="document.getElementById('bgMusic').pause()">⏸️ Pause</button>
        </div>
    """, unsafe_allow_html=True)
    
    time.sleep(5)
    st.session_state.first_load = False

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
                'Match_Status': 'Match',
                'New_HF_name_in_MFL': value1
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
                'Match_Status': 'Unmatch' if best_score < threshold else 'Match',
                'New_HF_name_in_MFL': best_match_row[col2] if best_score >= threshold else value1
            }
            # Add all columns from both dataframes
            for c in df1.columns:
                if c != col1:
                    result_row[f'MFL_{c}'] = row1[c]
            for c in df2.columns:
                if c != col2:
                    result_row[f'DHIS2_{c}'] = best_match_row[c] if best_match_row is not None else None
            results.append(result_row)
    
    # Add unmatched facilities from DHIS2
    for idx2, row2 in df2.iterrows():
        value2 = str(row2[col2])
        if value2 not in [str(r[f'DHIS2_{col2}']) for r in results]:
            result_row = {
                f'MFL_{col1}': None,
                f'DHIS2_{col2}': value2,
                'Match_Score': 0,
                'Match_Status': 'Unmatch',
                'New_HF_name_in_MFL': None
            }
            # Add all columns from both dataframes
            for c in df1.columns:
                if c != col1:
                    result_row[f'MFL_{c}'] = None
            for c in df2.columns:
                if c != col2:
                    result_row[f'DHIS2_{c}'] = row2[c]
            results.append(result_row)
    
    return pd.DataFrame(results)

def main():
    st.markdown('<h1 class="custom-title">Health Facility Name Matching</h1>', unsafe_allow_html=True)

    # Step 1: File Upload
    if st.session_state.step == 1:
        st.markdown('
