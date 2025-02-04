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

            st.markdown('<div class="section-header">Health Facilities Count</div>', unsafe_allow_html=True)
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

if __name__ == "__main__":
    main()
