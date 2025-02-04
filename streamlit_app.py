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
    

