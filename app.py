import streamlit as st
import os
from utils import retriever, prompt_builder, gemini_client, embeddings
from utils.sample_generator import generate_sample_data
from utils.xml_to_csv import process_xml_files

# Page Configuration
st.set_page_config(page_title="Healthcare RAG Assistant", page_icon="🩺", layout="wide")

# Custom CSS for chat interface
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    }
    .chat-message.user {
        background-color: #2b313e
    }
    .chat-message.bot {
        background-color: #475063
    }
    .chat-message .avatar {
        width: 15%;
    }
    .chat-message .message {
        width: 85%;
        padding: 0 1.5rem;
        color: #fff;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🩺 AI Healthcare Assistant")
    st.markdown("Ask medical questions and get trusted answers from the MedQuAD knowledge base.")

    # Sidebar: Configuration & Data Management
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key Management
        api_key = st.text_input("Google API Key", type="password", help="Enter your Gemini API Key if not in .env")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        
        # Data Management
        st.header("🗄️ Knowledge Base")
        
        if st.button("Generate Sample Data"):
            with st.spinner("Generating sample data..."):
                generate_sample_data()
                st.success("Sample data generated!")
                
        if st.button("Process XML Data"):
            with st.spinner("Converting XML to CSV..."):
                if process_xml_files():
                    st.success("XML processed successfully!")
                else:
                    st.warning("No XML files found in 'data/'")

        if st.button("Build/Rebuild Index"):
            with st.spinner("Embedding data and building index... (this may take a while)"):
                if embeddings.create_embeddings():
                    st.success("Index built successfully!")
                    # Reload resources in retriever
                    retriever._index = None 
                else:
                    st.error("Failed to build index. Ensure data exists.")

    # Main Chat Logic
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("What are the symptoms of flu?"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Searching medical records..."):
                # Configure Gemini
                try:
                    gemini_client.configure_gemini()
                except ValueError as e:
                    st.error(str(e))
                    st.stop()

                # 1. Retrieve
                results = retriever.retrieve(prompt)
                
                if not results:
                    response_text = "I couldn't find any relevant information in the database. Please try another question or ensure the database is indexed."
                else:
                    # 2. Build Prompt
                    llm_prompt = prompt_builder.build_prompt(prompt, results)
                    
                    # 3. Generate Answer
                    response_text = gemini_client.generate_answer(llm_prompt)

                st.markdown(response_text)
                
                # Show Sources
                if results:
                    with st.expander("📚 Trusted Sources"):
                        for i, res in enumerate(results):
                            st.markdown(f"**Source {i+1}:** {res['Focus']} (Score: {res['rerank_score']:.4f})")
                            st.write(f"*Q: {res['Question']}*")
                            st.caption(f"From: {res['Source']}")

        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    main()
