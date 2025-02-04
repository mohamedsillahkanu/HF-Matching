import streamlit as st

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
    }
}

def load_theme():
    """Initialize and load theme settings."""
    # Initialize theme state
    if 'theme_index' not in st.session_state:
        st.session_state.theme_index = list(themes.keys()).index("Black Modern")

    # Theme selection
    selected_theme = st.sidebar.selectbox(
        "🎨 Select Theme",
        list(themes.keys()),
        index=st.session_state.theme_index,
        key='theme_selector'
    )

    # Apply theme
    theme = themes[selected_theme]
    is_light_theme = "Light" in selected_theme
    theme_mode = "light" if is_light_theme else "dark"

    # Set theme variables
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
                --border-color: {'#DEE2E6' if is_light_theme else '#2E2
