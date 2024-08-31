import streamlit as st
from pages import web_search  # Import the chat_page function
from pages import doc_management  # Import the document_management_page function

def main():
    st.set_page_config(page_title="Pravaha", page_icon=":ocean:", layout="wide")

    # Add logo/image
    st.image("assets/pravha.png", width=200)  # Update the path to your logo

    st.title("Pravaha - Main Page")

    # Add content or navigation for your main page here
    st.write("Welcome to the main page of Pravaha. You can navigate to different pages using the sidebar.")

    # Sidebar navigation
    page = st.sidebar.selectbox("Select a page", ["Chat", "Document Management"])

    if page == "Chat":
        web_search.chat_page()
    elif page == "Document Management":
        doc_management.document_management_page()

if __name__ == "__main__":
    main()