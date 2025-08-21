import streamlit as st
import os
import yaml

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "../config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def load_frontend():
    cfg = load_config()
        # --------- Sidebar: Modes ---------
    with st.sidebar:
        st.header("Settings")
        mode = st.radio(
            "Mode",
            ["Interview", "Story", "FastFacts", "HumbleBrag", "Reflect"],
            index=0
        )
        st.caption("Mode tweaks tone & structure of answers.")


    # # --------- Main: Title + Resources ---------
    st.title("🗂️ Personal Codex Agent")
    st.caption("Ask about my work, skills, interests and values — grounded in my own docs.")

    with st.expander("🔗 Fun links about me"):
        res = cfg.get("resources", {})
        if res:
            if res.get("spotify_playlist"):
                st.markdown(f"- **Worship Spotify playlist:** [{res['spotify_playlist']}]({res['spotify_playlist']})")
            if res.get("chess_profile"):
                label = res.get("chess_username", res["chess_username"])
                st.markdown(f"- **Chess.com:** [{label}]({res['chess_profile']}) Make me a friend request ;)")
            if res.get("karate_achievements"):
                st.markdown(f"- **Karate achievements:** [{res['karate_achievements']}]({res['karate_achievements']})")
        else:
            st.info("Add links in `config.yaml → resources` to show them here.")

    return mode
