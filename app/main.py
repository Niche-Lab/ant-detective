import streamlit as st

# local imports
from sidebar import show_sidebar
from widgets import (
    image_uploader,
    show_navigator,
    show_download,
)
from globals import init_globals

st.set_page_config(
    page_title="Ant Detective",
    page_icon="🐜",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

def main():
    st.title("Ant Detective 🔎 🐜")
    init_globals()  
    image_uploader()
    
    show_download()        
    if st.session_state["loaded"]:
        detect_count = st.session_state["detect_count"]
        img = st.empty()        
        if detect_count == 0:
            st.error("No ants detected")
        else:
            show_navigator()
            cur_i = st.session_state["cur_i"]
            img.image(st.session_state["file_pred"][cur_i])   
    else:      
        # initial message
        st.success("Please upload images to get started")
                 
    # left-hand side
    show_sidebar()


main()
