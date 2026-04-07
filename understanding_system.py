import streamlit as st

def show_modal():
    if st.session_state.modal is None:
        return

    # overlay background (tetap pakai CSS)
    st.markdown("""
    <style>
    .modal-bg {
        position: fixed;
        top:0; left:0;
        width:100%; height:100%;
        background: rgba(0,0,0,0.6);
        z-index:9998;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="modal-bg"></div>', unsafe_allow_html=True)

    # 📦 MODAL CONTENT (PAKAI STREAMLIT, BUKAN HTML)
    with st.container():
        st.markdown("### 🌪️ Understanding the System")

        if st.session_state.modal == "cyclone":

            st.markdown("## Tropical Cyclone")

            st.write("""
            Tropical cyclones are large-scale atmospheric systems that develop over warm ocean waters,
            characterized by a low-pressure center, strong rotating winds, and organized convection.
            """)

            st.image(r"D:\Kuliah\Magang_BMKG\Project\Streamlit_cyclone\assets\siklon_tropis.jpg")

            st.write("""
            Tropical cyclones can trigger floods, landslides, storm surges, and extreme winds,
            causing major damage, displacement, and loss of life.
            """)

            st.image(r"D:\Kuliah\Magang_BMKG\Project\Streamlit_cyclone\assets\dampak_ST.jpeg")

        elif st.session_state.modal == "dvorak":
            st.markdown("## 🧠 Dvorak Technique")
            st.write("Penjelasan Dvorak nanti di sini...")

        elif st.session_state.modal == "deeplab":
            st.markdown("## ⚙️ DeepLabV3+")
            st.write("Penjelasan model segmentasi...")

        elif st.session_state.modal == "localization":
            st.markdown("## 📍 Localization")
            st.write("Penjelasan bounding box...")

        elif st.session_state.modal == "llm":
            st.markdown("## 💬 LLM Integration")
            st.write("Penjelasan Grok...")

        st.markdown("---")

        if st.button("❌ Close"):
            st.session_state.modal = None