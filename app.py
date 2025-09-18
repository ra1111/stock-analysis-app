import streamlit as st
import pandas as pd
import numpy as np
from tabs import portfolio_tab, ipo_tab, screener_tab, optimizer_tab, news_tab

# Configure the page
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("📈 Stock Analysis Dashboard")
    st.markdown("---")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Portfolio", 
        "🚀 IPO Radar", 
        "🔍 Stock Screener", 
        "⚖️ Portfolio Optimizer", 
        "📰 News Analysis"
    ])
    
    with tab1:
        portfolio_tab.render()
    
    with tab2:
        ipo_tab.render()
    
    with tab3:
        screener_tab.render()
    
    with tab4:
        optimizer_tab.render()
    
    with tab5:
        news_tab.render()

if __name__ == "__main__":
    main()
