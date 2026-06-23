import streamlit as st

pg = st.navigation(
    [
        st.Page("pgs/home.py", title="Home", icon="🏠"),
        st.Page("pgs/upload.py", title="Upload File", icon="🗳️"),
        st.Page("pgs/upload_chunks.py", title="Upload Chunks", icon="🧩"),
        st.Page("pgs/chat.py", title="Chat with Graph" , icon="🦜"),
    ]
)

pg.run()
