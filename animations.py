import streamlit as st
import random
import time
import threading

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

def init_animations():
    """Initialize animations in the session state."""
    if 'last_animation' not in st.session_state:
        st.session_state.last_animation = time.time()
        st.session_state.first_load = True

    if st.session_state.first_load:
        st.balloons()
        st.snow()
        welcome_placeholder = st.empty()
        welcome_placeholder.success("Welcome! 🌟")
        time.sleep(3)
        welcome_placeholder.empty()
        st.session_state.first_load = False

def start_animation_thread():
    """Start the periodic animation thread."""
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
