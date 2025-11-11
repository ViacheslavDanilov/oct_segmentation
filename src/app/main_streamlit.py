import os
from glob import glob

import streamlit as st

from src.app.tools.analysis import get_analysis, get_gpt_analysis
from src.app.tools.img_viewer import get_img_show
from src.app.tools.plotly_analytics import get_plot_area, get_trace_area
from src.data.utils import CLASS_IDS


def get_patient_files():
    """Get list of available patient files."""
    patient_dir = "data/app/demo/patients"
    files = sorted(glob(os.path.join(patient_dir, "*.npz")))
    return files


def get_info_metric(name: str, value: str, description: str):
    """Create a styled metric display."""
    return f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 4px rgba(102, 126, 234, 0.2);
        height: 100%;
        color: white;
        transition: transform 0.3s ease;
    ">
        <div style="font-size: 16px; font-weight: 500; opacity: 0.95; margin-bottom: 8px;">{name}</div>
        <div style="font-size: 32px; font-weight: 700; margin: 12px 0; letter-spacing: -0.5px;">{value}</div>
        <div style="font-size: 14px; opacity: 0.9;">{description}</div>
    </div>
    """


def initialize_session_state():
    """Initialize session state variables."""
    if "data" not in st.session_state:
        st.session_state.data = None
    if "images" not in st.session_state:
        st.session_state.images = None
    if "slices" not in st.session_state:
        st.session_state.slices = 0
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False
    if "analysis_summary" not in st.session_state:
        st.session_state.analysis_summary = ""
    if "risk_panel" not in st.session_state:
        st.session_state.risk_panel = ""


def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="OCT Analysis",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for modern styling
    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(to bottom, #f8f9fa 0%, #ffffff 100%);
        }
        .stApp {
            max-width: 100%;
        }
        h1 {
            color: #2E86DE;
            font-weight: 700;
            padding-bottom: 20px;
            border-bottom: 3px solid #E41EC7;
            margin-bottom: 30px;
        }
        h2 {
            color: #667eea;
            font-weight: 600;
            margin-top: 30px;
            margin-bottom: 20px;
        }
        h3 {
            color: #764ba2;
            font-weight: 500;
        }
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 12px 32px;
            font-weight: 600;
            font-size: 16px;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
            transition: all 0.3s ease;
            width: 100%;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(102, 126, 234, 0.5);
        }
        .stSelectbox, .stMultiselect {
            border-radius: 8px;
        }
        .stSlider {
            padding: 10px 0;
        }
        div[data-testid="stMetricValue"] {
            font-size: 28px;
            color: #2E86DE;
            font-weight: 700;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 14px;
            color: #666;
            font-weight: 500;
        }
        .stPlotlyChart {
            border-radius: 16px;
            box-shadow: 0 2px 16px rgba(102, 126, 234, 0.15);
            background: white;
            padding: 20px;
            border: 1px solid rgba(102, 126, 234, 0.2);
            overflow: hidden;
        }
        .stPlotlyChart > div {
            border-radius: 12px;
        }
        .js-plotly-plot .plotly {
            border-radius: 12px;
        }
        div.block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .sidebar .sidebar-content {
            background: linear-gradient(to bottom, #ffffff 0%, #f8f9fa 100%);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    initialize_session_state()

    # Header
    st.markdown("# 🔬 Optical Coherence Tomography Analysis")

    # Sidebar for controls
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")

        # Patient file selection
        patient_files = get_patient_files()
        if patient_files:
            selected_file = st.selectbox(
                "📂 Patient Data",
                options=patient_files,
                index=0,
                help="Select patient file for analysis",
            )
        else:
            st.error("❌ Patient files not found")
            st.stop()

        st.markdown("---")

        # Analysis button
        if st.button("🚀 Run Analysis", use_container_width=True):
            with st.spinner("🔄 Processing data... Please wait"):
                try:
                    # Run analysis
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Load and process data
                    status_text.text("Loading data...")
                    progress_bar.progress(20)

                    result = get_analysis(selected_file, "demo")

                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")

                    # Store results in session state
                    st.session_state.slices = result[0].maximum
                    st.session_state.data = result[7]
                    st.session_state.images = result[8]
                    st.session_state.analysis_done = True
                    st.session_state.mean_area_lumen = result[9]
                    st.session_state.min_size_fc = result[10]
                    st.session_state.mean_size_fc = result[11]
                    st.session_state.counter_fc = result[12]

                    try:
                        summary, risk_panel = get_gpt_analysis(st.session_state.data)
                        st.session_state.analysis_summary = summary
                        st.session_state.risk_panel = risk_panel
                    except Exception as err:  # pragma: no cover - defensive guard
                        st.session_state.analysis_summary = ""
                        st.session_state.risk_panel = ""
                        st.warning(f"Risk analysis unavailable: {err}")

                    st.success("✅ Analysis completed successfully!")
                    progress_bar.empty()
                    status_text.empty()
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Analysis error: {str(e)}")
                    st.stop()

        st.markdown("---")

        # Visualization controls (only show after analysis)
        if st.session_state.analysis_done:
            st.markdown("### 🎨 Visualization Settings")

            # Frame selector
            frame_num = st.slider(
                "🎞️ Frame Number",
                min_value=0,
                max_value=st.session_state.slices,
                value=0,
                help="Select frame to view",
            )

            # Class selection
            selected_classes = st.multiselect(
                "🎯 Objects",
                options=list(CLASS_IDS.keys()),
                default=list(CLASS_IDS.keys()),
                help="Select objects to display",
            )

            # Transparency control
            transparency = st.slider(
                "👁️ Transparency, %",
                min_value=0,
                max_value=100,
                value=20,
                help="Adjust mask transparency",
            )

            # Chart class selection
            st.markdown("---")
            st.markdown("### 📊 Chart Settings")

            classes_for_charts = st.multiselect(
                "📈 Objects for Charts",
                options=list(CLASS_IDS.keys()),
                default=list(CLASS_IDS.keys()),
                help="Select objects to display in charts",
            )

    # Main content area
    if st.session_state.analysis_done:
        st.markdown("## 🩺 Risk & Insight")
        
        # Create custom HTML/CSS for proper column spanning
        risk_html = st.session_state.risk_panel if st.session_state.risk_panel else """
            <div style="
                background: linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%);
                border-radius: 16px;
                padding: 24px;
                text-align: center;
                box-shadow: 0 4px 4px rgba(127, 140, 141, 0.2);
                height: 100%;
                color: white;
            ">
                <div style="font-size: 16px; font-weight: 500; opacity: 0.95; margin-bottom: 8px;">Risk</div>
                <div style="font-size: 32px; font-weight: 700; margin: 12px 0;">—</div>
                <div style="font-size: 14px; opacity: 0.9;">Unavailable</div>
            </div>
        """
        
        if st.session_state.analysis_summary:
            summary_html = f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 16px;
                    padding: 24px;
                    box-shadow: 0 4px 4px rgba(102, 126, 234, 0.2);
                    height: 100%;
                    color: white;
                ">
                    <div style="font-size: 16px; font-weight: 500; opacity: 0.95; margin-bottom: 12px;">Analysis Result</div>
                    <div style="font-size: 14px; line-height: 1.6; opacity: 0.95; max-height: 180px; overflow-y: auto;">
                        {st.session_state.analysis_summary}
                    </div>
                </div>
            """
        else:
            summary_html = """
                <div style="
                    background: linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%);
                    border-radius: 16px;
                    padding: 24px;
                    box-shadow: 0 4px 4px rgba(127, 140, 141, 0.2);
                    height: 100%;
                    color: white;
                ">
                    <div style="font-size: 16px; font-weight: 500; opacity: 0.95; margin-bottom: 12px;">Analysis Result</div>
                    <div style="font-size: 14px; line-height: 1.6; opacity: 0.95;">
                        Processing...
                    </div>
                </div>
            """
        
        # Use CSS Grid to create the layout: 1 column for risk, 3 columns for summary
        st.markdown(
            f"""
            <div style="display: grid; grid-template-columns: 1fr 3fr; gap: 16px; margin-bottom: 20px;">
                <div>{risk_html}</div>
                <div>{summary_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Metrics section
        st.markdown("## 📊 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(st.session_state.mean_area_lumen, unsafe_allow_html=True)
        with col2:
            st.markdown(st.session_state.min_size_fc, unsafe_allow_html=True)
        with col3:
            st.markdown(st.session_state.mean_size_fc, unsafe_allow_html=True)
        with col4:
            st.markdown(st.session_state.counter_fc, unsafe_allow_html=True)

        st.markdown("---")

        # Image viewer section
        st.markdown("## 🖼️ Image Visualization")

        # Display image with masks
        if st.session_state.data and st.session_state.images:
            fig = get_img_show(
                data=st.session_state.data,
                images=st.session_state.images,
                img_num=frame_num,
                classes_vis=selected_classes,
                opacity=transparency,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Charts section
        col_chart1, col_chart2 = st.columns(2, gap="medium")

        with col_chart1:
            st.markdown("### 📈 Area Dynamics")
            if st.session_state.data:
                fig_trace = get_trace_area(
                    classes=classes_for_charts,
                    data=st.session_state.data,
                )
                st.plotly_chart(fig_trace, use_container_width=True)

        with col_chart2:
            st.markdown("### 📊 Area Distribution")
            if st.session_state.data:
                fig_box = get_plot_area(
                    classes=classes_for_charts,
                    data=st.session_state.data,
                )
                st.plotly_chart(fig_box, use_container_width=True)

    else:
        # Welcome screen
        st.markdown(
            """
            <div style="
                text-align: center;
                padding: 80px 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 20px;
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
                color: white;
                margin: 40px 0;
            ">
                <h2 style="color: white; font-size: 42px; margin-bottom: 20px;">
                    👋 Welcome to OCT Analysis System
                </h2>
                <p style="font-size: 20px; opacity: 0.95; max-width: 700px; margin: 0 auto; line-height: 1.6;">
                    Select a patient file from the sidebar and click <strong>"Run Analysis"</strong>
                    to begin processing optical coherence tomography data.
                </p>
                <div style="margin-top: 40px; font-size: 16px; opacity: 0.9;">
                    <p>🔬 Tissue structure analysis</p>
                    <p>📊 Detailed visualization and statistics</p>
                    <p>🎯 High-precision object segmentation</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 20px;">
            <p style="margin: 0; font-size: 14px;">
                🔬 Optical Coherence Tomography Analysis System | Powered by Streamlit
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
