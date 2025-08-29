"""
Professional Styling System for Spend Platform
Enterprise-grade CSS and layout utilities using Tailwind-inspired design principles
"""

import streamlit as st

# Color scheme constants
COLORS = {
    'primary': {
        'dark': '#1e293b',      # slate-800
        'light': '#f1f5f9',     # slate-100
        'medium': '#475569',    # slate-600
    },
    'secondary': {
        'lightest': '#f9fafb',  # gray-50
        'light': '#f3f4f6',     # gray-100
        'medium': '#6b7280',    # gray-500
        'dark': '#111827',      # gray-900
    },
    'accent': {
        'blue': '#2563eb',      # blue-600
        'blue_light': '#3b82f6', # blue-500
        'blue_dark': '#1d4ed8',  # blue-700
    },
    'status': {
        'success': '#10b981',   # emerald-500
        'warning': '#f59e0b',   # amber-500
        'error': '#ef4444',     # red-500
        'info': '#06b6d4',      # cyan-500
    }
}

def inject_custom_css():
    """Inject comprehensive CSS styling for professional appearance"""
    css = f"""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global CSS Variables */
    :root {{
        --primary-dark: {COLORS['primary']['dark']};
        --primary-light: {COLORS['primary']['light']};
        --primary-medium: {COLORS['primary']['medium']};
        --secondary-lightest: {COLORS['secondary']['lightest']};
        --secondary-light: {COLORS['secondary']['light']};
        --secondary-medium: {COLORS['secondary']['medium']};
        --secondary-dark: {COLORS['secondary']['dark']};
        --accent-blue: {COLORS['accent']['blue']};
        --accent-blue-light: {COLORS['accent']['blue_light']};
        --accent-blue-dark: {COLORS['accent']['blue_dark']};
        --success: {COLORS['status']['success']};
        --warning: {COLORS['status']['warning']};
        --error: {COLORS['status']['error']};
        --info: {COLORS['status']['info']};
        --border-radius: 0.5rem;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    
    /* Global Styles */
    .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        background-color: var(--secondary-lightest);
        color: var(--secondary-dark);
    }}
    
    /* Remove default Streamlit styling */
    .css-1d391kg {{
        padding: 0;
    }}
    
    /* Header Styling */
    .main-header {{
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary-medium) 100%);
        padding: 1rem 1.5rem;
        color: white;
        margin-bottom: 1.5rem;
        border-radius: 0 0 0.75rem 0.75rem;
        box-shadow: var(--shadow-lg);
    }}
    
    .main-header h1 {{
        margin: 0;
        font-size: 1.5rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    .main-header .subtitle {{
        margin: 0.25rem 0 0 0;
        font-size: 0.875rem;
        opacity: 0.9;
        font-weight: 400;
    }}
    
    /* Sidebar Styling */
    .css-1d391kg .css-1y4p8pa {{
        background-color: white;
        border-right: 1px solid var(--secondary-light);
        box-shadow: var(--shadow-md);
    }}
    
    .sidebar-section {{
        background: white;
        border-radius: var(--border-radius);
        padding: 0.75rem;
        margin-bottom: 0.75rem;
        border: 1px solid var(--secondary-light);
        box-shadow: var(--shadow-sm);
    }}
    
    .sidebar-section h2 {{
        color: var(--primary-dark);
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        padding-bottom: 0.25rem;
        border-bottom: 2px solid var(--accent-blue);
    }}
    
    /* Card Components */
    .metric-card {{
        background: white;
        border-radius: var(--border-radius);
        padding: 1rem;
        border: 1px solid var(--secondary-light);
        box-shadow: var(--shadow-sm);
        transition: var(--transition);
        margin-bottom: 1rem;
    }}
    
    .metric-card:hover {{
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }}
    
    .metric-value {{
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--primary-dark);
        margin-bottom: 0.25rem;
    }}
    
    .metric-label {{
        font-size: 0.75rem;
        color: var(--secondary-medium);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }}
    
    .metric-delta {{
        font-size: 0.75rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }}
    
    .metric-delta.positive {{
        color: var(--success);
    }}
    
    .metric-delta.negative {{
        color: var(--error);
    }}
    
    /* Button Styling */
    .stButton > button {{
        background-color: var(--accent-blue);
        color: white;
        border: none;
        border-radius: var(--border-radius);
        padding: 0.5rem 1rem;
        font-weight: 500;
        font-size: 0.8rem;
        transition: var(--transition);
        box-shadow: var(--shadow-sm);
    }}
    
    .stButton > button:hover {{
        background-color: var(--accent-blue-dark);
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }}
    
    .stButton > button:active {{
        transform: translateY(0);
        box-shadow: var(--shadow-sm);
    }}
    
    /* Secondary Button */
    .btn-secondary {{
        background-color: white !important;
        color: var(--primary-dark) !important;
        border: 1px solid var(--secondary-light) !important;
    }}
    
    .btn-secondary:hover {{
        background-color: var(--secondary-light) !important;
        border-color: var(--secondary-medium) !important;
    }}
    
    /* Success Button */
    .btn-success {{
        background-color: var(--success) !important;
    }}
    
    .btn-success:hover {{
        background-color: #059669 !important;
    }}
    
    /* Warning Button */
    .btn-warning {{
        background-color: var(--warning) !important;
    }}
    
    .btn-warning:hover {{
        background-color: #d97706 !important;
    }}
    
    /* Danger Button */
    .btn-danger {{
        background-color: var(--error) !important;
    }}
    
    .btn-danger:hover {{
        background-color: #dc2626 !important;
    }}
    
    /* Input Styling */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextArea > div > div > textarea {{
        border: 1px solid var(--secondary-light);
        border-radius: var(--border-radius);
        padding: 0.5rem 0.75rem;
        background-color: white !important;
        color: var(--secondary-dark) !important;
        transition: var(--transition);
        font-size: 0.875rem;
        line-height: 1.4;
    }}
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stTextArea > div > div > textarea:focus {{
        border-color: var(--accent-blue);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        outline: none;
        background-color: white !important;
        color: var(--secondary-dark) !important;
    }}
    
    /* Fix input placeholder text */
    .stTextInput > div > div > input::placeholder {{
        color: var(--secondary-medium) !important;
        opacity: 0.7;
    }}
    
    /* Fix select box text */
    .stSelectbox > div > div > select option {{
        color: var(--secondary-dark) !important;
        background-color: white !important;
    }}
    
    /* DataFrame Styling */
    .dataframe {{
        background: white;
        border-radius: var(--border-radius);
        border: 1px solid var(--secondary-light);
        box-shadow: var(--shadow-sm);
        overflow: hidden;
    }}
    
    .dataframe table {{
        font-size: 0.875rem;
    }}
    
    .dataframe th {{
        background-color: var(--secondary-light);
        color: var(--primary-dark);
        font-weight: 600;
        padding: 1rem 0.75rem;
        border-bottom: 2px solid var(--secondary-medium);
    }}
    
    .dataframe td {{
        padding: 0.75rem;
        border-bottom: 1px solid var(--secondary-light);
    }}
    
    .dataframe tbody tr:hover {{
        background-color: var(--secondary-lightest);
    }}
    
    /* Alert/Message Styling */
    .stAlert {{
        border-radius: var(--border-radius);
        border: none;
        box-shadow: var(--shadow-sm);
    }}
    
    .stSuccess {{
        background-color: #f0fdf4;
        color: #166534;
        border-left: 4px solid var(--success);
    }}
    
    .stError {{
        background-color: #fef2f2;
        color: #991b1b;
        border-left: 4px solid var(--error);
    }}
    
    .stWarning {{
        background-color: #fffbeb;
        color: #92400e;
        border-left: 4px solid var(--warning);
    }}
    
    .stInfo {{
        background-color: #f0f9ff;
        color: #0c4a6e;
        border-left: 4px solid var(--info);
    }}
    
    /* Expander Styling */
    .streamlit-expanderHeader {{
        background-color: white;
        border: 1px solid var(--secondary-light);
        border-radius: var(--border-radius);
        padding: 1rem;
        font-weight: 500;
        color: var(--primary-dark);
    }}
    
    .streamlit-expanderHeader:hover {{
        background-color: var(--secondary-lightest);
    }}
    
    .streamlit-expanderContent {{
        background-color: white;
        border: 1px solid var(--secondary-light);
        border-top: none;
        border-radius: 0 0 var(--border-radius) var(--border-radius);
        padding: 1rem;
    }}
    
    /* Tab Styling */
    .stTabs {{
        background-color: white;
        border-radius: var(--border-radius);
        border: 1px solid var(--secondary-light);
        box-shadow: var(--shadow-sm);
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        background-color: var(--secondary-light);
        padding: 0.25rem;
        border-radius: var(--border-radius) var(--border-radius) 0 0;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        color: var(--secondary-medium);
        border-radius: calc(var(--border-radius) * 0.75);
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        margin: 0.25rem;
        transition: var(--transition);
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: white;
        color: var(--primary-dark);
        box-shadow: var(--shadow-sm);
    }}
    
    /* Chart Container */
    .chart-container {{
        background: white;
        border-radius: var(--border-radius);
        padding: 1.5rem;
        border: 1px solid var(--secondary-light);
        box-shadow: var(--shadow-sm);
        margin-bottom: 1.5rem;
    }}
    
    /* Loading States */
    .stSpinner {{
        border-color: var(--accent-blue) !important;
    }}
    
    /* Custom Classes */
    .section-header {{
        color: var(--primary-dark);
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        padding-bottom: 0.25rem;
        border-bottom: 2px solid var(--accent-blue);
    }}
    
    .subsection-header {{
        color: var(--primary-dark);
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        margin-top: 1rem;
    }}
    
    .status-badge {{
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }}
    
    .status-active {{
        background-color: #dcfce7;
        color: #166534;
    }}
    
    .status-inactive {{
        background-color: #fee2e2;
        color: #991b1b;
    }}
    
    .status-pending {{
        background-color: #fef3c7;
        color: #92400e;
    }}
    
    /* Responsive Design */
    @media (max-width: 768px) {{
        .main-header {{
            padding: 0.75rem;
        }}
        
        .main-header h1 {{
            font-size: 1.25rem;
        }}
        
        .metric-card {{
            padding: 0.75rem;
        }}
        
        .metric-value {{
            font-size: 1.25rem;
        }}
    }}
    
    /* Dark Mode Support */
    @media (prefers-color-scheme: dark) {{
        :root {{
            --secondary-lightest: #1f2937;
            --secondary-light: #374151;
            --secondary-medium: #9ca3af;
            --secondary-dark: #f9fafb;
        }}
        
        .stApp {{
            background-color: var(--secondary-lightest);
            color: var(--secondary-dark);
        }}
        
        .metric-card,
        .sidebar-section,
        .dataframe,
        .chart-container {{
            background-color: #374151;
            border-color: #4b5563;
        }}
    }}
    
    /* Animation Classes */
    .fade-in {{
        animation: fadeIn 0.3s ease-in;
    }}
    
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .slide-in {{
        animation: slideIn 0.3s ease-out;
    }}
    
    @keyframes slideIn {{
        from {{ opacity: 0; transform: translateX(-10px); }}
        to {{ opacity: 1; transform: translateX(0); }}
    }}
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)

def create_metric_card(title, value, delta=None, delta_positive=True):
    """Create a professional metric card component"""
    delta_class = "positive" if delta_positive else "negative"
    delta_html = f'<div class="metric-delta {delta_class}">{delta}</div>' if delta else ""
    
    html = f"""
    <div class="metric-card fade-in">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{title}</div>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def create_section_header(title, subtitle=None):
    """Create a professional section header"""
    subtitle_html = f'<p class="subtitle">{subtitle}</p>' if subtitle else ""
    html = f"""
    <div class="main-header">
        <h1>{title}</h1>
        {subtitle_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def create_status_badge(status, text=None):
    """Create a status badge with appropriate styling"""
    text = text or status
    class_name = f"status-{status.lower()}"
    html = f'<span class="status-badge {class_name}">{text}</span>'
    return html

def create_info_box(title, content, type="info"):
    """Create an information box with icon and content"""
    icons = {
        "info": "ℹ️",
        "success": "✅", 
        "warning": "⚠️",
        "error": "❌"
    }
    
    html = f"""
    <div class="info-box {type}">
        <div class="info-box-header">
            <span class="info-icon">{icons.get(type, "ℹ️")}</span>
            <strong>{title}</strong>
        </div>
        <div class="info-box-content">{content}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def apply_button_style(button_type="primary"):
    """Apply custom button styling"""
    if button_type == "secondary":
        st.markdown('<style>.stButton > button { background-color: white !important; color: var(--primary-dark) !important; border: 1px solid var(--secondary-light) !important; }</style>', unsafe_allow_html=True)
    elif button_type == "success":
        st.markdown('<style>.stButton > button { background-color: var(--success) !important; }</style>', unsafe_allow_html=True)
    elif button_type == "warning":
        st.markdown('<style>.stButton > button { background-color: var(--warning) !important; }</style>', unsafe_allow_html=True)
    elif button_type == "danger":
        st.markdown('<style>.stButton > button { background-color: var(--error) !important; }</style>', unsafe_allow_html=True)

def create_professional_dataframe(df, title=None):
    """Display a dataframe with professional styling"""
    if title:
        st.markdown(f'<h3 class="subsection-header">{title}</h3>', unsafe_allow_html=True)
    
    # Wrap dataframe in styled container
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def create_chart_container(chart_func, title=None):
    """Wrap charts in professional container"""
    if title:
        st.markdown(f'<h3 class="subsection-header">{title}</h3>', unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    chart_func()
    st.markdown('</div>', unsafe_allow_html=True)

def create_coming_soon_page(page_title, page_icon, features_list, timeline="Q4 2025"):
    """Create a professional coming soon page with feature preview"""
    create_section_header(
        f"{page_icon} {page_title}", 
        "This feature is currently under development"
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 🚧 Development Status")
        st.markdown(f"""
        **Status**: In Development  
        **Expected Release**: {timeline}  
        **Priority**: High  
        
        This feature is part of our comprehensive spend management platform expansion. 
        We're working hard to bring you the best possible experience.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.markdown("### 🎯 Planned Features")
        for feature in features_list:
            st.markdown(f"✨ {feature}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 📧 Get Notified")
        st.markdown("""
        Want to be notified when this feature is ready?
        
        👥 **Beta Testing Program**  
        Join our beta testing program to get early access to new features.
        """)
        
        if st.button("🔔 Request Access", use_container_width=True):
            st.success("✅ You'll be notified when this feature is available!")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Progress indicator
        st.markdown('<div class="metric-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.markdown("### 📊 Development Progress")
        st.progress(0.65)  # 65% complete
        st.markdown("*65% Complete*")
        st.markdown('</div>', unsafe_allow_html=True)
