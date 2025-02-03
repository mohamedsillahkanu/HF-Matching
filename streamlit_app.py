import streamlit as st
import pandas as pd
import numpy as np
from jellyfish import jaro_winkler_similarity
from io import BytesIO
from PIL import Image

st.image("https://raw.githubusercontent.com/mohamedsillahkanu/si/4a7954a650c20056f19da5a77fe5d09ed4e49526/welcome-animation.svg", caption="Welcome Animation")

def calculate_match(df1, df2, col1, col2, threshold):
    """Calculate matching scores between two columns using Jaro-Winkler similarity.
    Returns a DataFrame with all columns from both input DataFrames."""
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
    
    # Add unmatched facilities from DHIS2
    for idx2, row2 in df2.iterrows():
        value2 = str(row2[col2])
        if value2 not in [str(r[f'DHIS2_{col2}']) for r in results]:
            result_row = {
                f'MFL_{col1}': None,
                f'DHIS2_{col2}': value2,
                'Match_Score': 0,
                'Match_Status': 'Unmatch'
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
    st.title("Health Facility Name Matching")

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
            
            master_hf_list_clean[mfl_col] = master_hf_list_clean[mfl_col].astype(str)
            master_hf_list_clean = master_hf_list_clean.drop_duplicates(subset=[mfl_col])
            dhis2_list_clean[dhis2_col] = dhis2_list_clean[dhis2_col].astype(str)

            st.write("### Counts of Health Facilities")
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

                # Display results
                st.write("### Matching Results")
                st.dataframe(hf_name_match_results)

                # Calculate and display summary statistics
                total_facilities = len(hf_name_match_results)
                matched_facilities = len(hf_name_match_results[hf_name_match_results['Match_Status'] == 'Match'])
                unmatched_facilities = total_facilities - matched_facilities
                
                st.write("### Summary Statistics")
                st.write(f"Total Facilities: {total_facilities}")
                st.write(f"Matched Facilities: {matched_facilities}")
                st.write(f"Unmatched Facilities: {unmatched_facilities}")
                st.write(f"Match Rate: {(matched_facilities/total_facilities*100):.2f}%")

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

if __name__ == "__main__":
    main()
