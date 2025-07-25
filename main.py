import streamlit as st
from rag import query_from_urls

st.set_page_config(page_title="🔎 Web Q&A with Groq", page_icon="🧠", layout="centered")

st.title("🔎 Web Q&A")
st.markdown("Ask questions from any website. Powered by **LLaMA3 + Chroma + HuggingFace** 💡")

st.markdown("#### 🌐 Enter URLs (comma-separated)")
url_input = st.text_input("Webpage URLs", placeholder="https://example.com, https://another.com")

question = st.text_input("💬 Your Question", placeholder="What is this page about?")

if st.button("Ask"):
    if not url_input or not question:
        st.warning("Please enter both the URLs and a question.")
    else:
        urls = [u.strip() for u in url_input.split(",") if u.strip()]
        with st.spinner("Processing and thinking... 🤖"):
            try:
                answer, used_urls = query_from_urls(urls, question)
                st.success("✅ Answer:")
                st.markdown(f"> {answer}")

                st.markdown("---")
                st.markdown("**🔗 Sources:**")
                for u in used_urls:
                    st.markdown(f"- [{u}]({u})")
            except Exception as e:
                st.error(f"❌ Error: {e}")
