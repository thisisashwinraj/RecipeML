import streamlit as st


class Theme:
    def __init__(self):
        if "themes" not in st.session_state: 
            st.session_state.themes = {"current_theme": "light",
                            "refreshed": True,
                            
                            "light": {"theme.base": "dark",
                                    "theme.backgroundColor": "#111111",
                                    "theme.primaryColor": "#64ABD8",
                                    "theme.secondaryBackgroundColor": "#181818",
                                    "theme.textColor": "#FFFFFF",
                                    "button_face": "🌜"},

                            "dark":  {"theme.base": "light",
                                    "theme.backgroundColor": "blue",
                                    "theme.primaryColor": "#64ABD8",
                                    "theme.secondaryBackgroundColor": "black",
                                    "theme.textColor": "black",
                                    "button_face": "☀️"},
                            }

    def change_theme(self):
        previous_theme = st.session_state.themes["current_theme"]
        tdict = st.session_state.themes["light"] if st.session_state.themes["current_theme"] == "light" else st.session_state.themes["dark"]
        for vkey, vval in tdict.items(): 
            if vkey.startswith("theme"): st._config.set_option(vkey, vval)

        st.session_state.themes["refreshed"] = False

        if previous_theme == "dark": st.session_state.themes["current_theme"] = "light"
        elif previous_theme == "light": st.session_state.themes["current_theme"] = "dark"

        if st.session_state.themes["refreshed"] == False:
            st.session_state.themes["refreshed"] = True
            st.rerun()
