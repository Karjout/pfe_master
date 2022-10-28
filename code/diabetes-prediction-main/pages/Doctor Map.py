import streamlit as st
import leafmap.foliumap as leafmap


def main():
    st.title("Doctor Map")

    st.markdown(
        """
    
    """
    )

    m = leafmap.Map(locate_control=True)
    m.add_basemap("ROADMAP")
    m.to_streamlit(height=700)


if __name__ == "__main__":
    main()
