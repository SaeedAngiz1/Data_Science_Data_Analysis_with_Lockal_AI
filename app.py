import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO, BytesIO
import requests
import json
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="AI-Powered Data Analysis Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'processing_history' not in st.session_state:
    st.session_state.processing_history = []

# Ollama API Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

def call_ollama(prompt):
    """Call Ollama API for AI assistance"""
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json().get("response", "No response from LLM")
        else:
            return f"Error: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"Connection error: Make sure Ollama is running on localhost:11434. Error: {str(e)}"

def add_to_history(action, details):
    """Add action to processing history"""
    st.session_state.processing_history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "details": details
    })

# Main Header
st.markdown('<div class="main-header">📊 AI-Powered Data Analysis Suite</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home & Upload", "🔍 Data Overview", "🧹 Data Cleaning", "🔢 Null Analysis", "⚙️ Data Processing", "📈 Visualizations", "🤖 AI Assistant", "💾 Export Data"]
)

# Home & Upload Page
if page == "🏠 Home & Upload":
    st.header("Welcome to Data Analysis Suite")
    st.markdown("""
    This app provides comprehensive data analysis capabilities:
    - **Data Upload**: Support for CSV, Excel, JSON files
    - **Data Cleaning**: Remove duplicates, handle outliers, type conversion
    - **Null Analysis**: Detect and handle missing values
    - **Data Processing**: Transformations and feature engineering
    - **Visualizations**: Interactive charts and graphs
    - **AI Assistant**: Get help from local LLM (Ollama)
    - **Export**: Download processed data
    """)
    
    st.subheader("Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls', 'json'],
        help="Upload CSV, Excel, or JSON files"
    )
    
    if uploaded_file is not None:
        try:
            # Load data based on file type
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            else:
                st.error("Unsupported file format")
                df = None
            
            if df is not None:
                st.session_state.df = df.copy()
                st.session_state.df_processed = df.copy()
                st.success(f"✅ File uploaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                add_to_history("Data Upload", f"Uploaded {uploaded_file.name} with shape {df.shape}")
                
                # Quick preview
                st.subheader("Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    # Ollama connection check
    st.sidebar.subheader("AI Assistant Status")
    if st.sidebar.button("Check Ollama Connection"):
        test_response = call_ollama("Say 'Connected' if you can read this.")
        if "Connected" in test_response or "connected" in test_response:
            st.sidebar.success("✅ Ollama is connected!")
        else:
            st.sidebar.warning("⚠️ Ollama might not be running. Check your connection.")

# Data Overview Page
elif page == "🔍 Data Overview":
    if st.session_state.df is not None:
        df = st.session_state.df_processed if st.session_state.df_processed is not None else st.session_state.df
        
        st.header("Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        with col4:
            st.metric("Null Values", df.isnull().sum().sum())
        
        st.subheader("Dataset Info")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(col_info, use_container_width=True)
        
        st.subheader("Full Dataset")
        st.dataframe(df, use_container_width=True)
        
    else:
        st.warning("⚠️ Please upload a dataset first from the Home page.")

# Data Cleaning Page
elif page == "🧹 Data Cleaning":
    if st.session_state.df is not None:
        st.header("Data Cleaning")
        df = st.session_state.df_processed if st.session_state.df_processed is not None else st.session_state.df
        
        st.subheader("Current Dataset Shape")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        
        # Remove Duplicates
        st.subheader("Remove Duplicates")
        if st.button("Remove Duplicate Rows"):
            initial_shape = df.shape[0]
            df = df.drop_duplicates()
            removed = initial_shape - df.shape[0]
            st.session_state.df_processed = df.copy()
            st.success(f"✅ Removed {removed} duplicate rows. New shape: {df.shape}")
            add_to_history("Remove Duplicates", f"Removed {removed} duplicate rows")
        
        # Handle Outliers using IQR method
        st.subheader("Handle Outliers")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Select Column for Outlier Removal", numeric_cols)
            col1, col2 = st.columns(2)
            with col1:
                outlier_method = st.radio("Method", ["IQR", "Z-Score"])
            with col2:
                action = st.radio("Action", ["Remove", "Cap"])
            
            if st.button("Apply Outlier Treatment"):
                initial_shape = df.shape[0]
                if outlier_method == "IQR":
                    Q1 = df[selected_col].quantile(0.25)
                    Q3 = df[selected_col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    if action == "Remove":
                        df = df[(df[selected_col] >= lower_bound) & (df[selected_col] <= upper_bound)]
                    else:  # Cap
                        df[selected_col] = df[selected_col].clip(lower_bound, upper_bound)
                else:  # Z-Score
                    z_scores = np.abs((df[selected_col] - df[selected_col].mean()) / df[selected_col].std())
                    if action == "Remove":
                        df = df[z_scores < 3]
                    else:  # Cap at 3 standard deviations
                        mean = df[selected_col].mean()
                        std = df[selected_col].std()
                        df[selected_col] = df[selected_col].clip(mean - 3*std, mean + 3*std)
                
                removed = initial_shape - df.shape[0]
                st.session_state.df_processed = df.copy()
                st.success(f"✅ Outliers treated. Removed: {removed} rows" if action == "Remove" else f"✅ Outliers capped.")
                add_to_history("Outlier Treatment", f"{outlier_method} method on {selected_col}, {action}")
        
        # Data Type Conversion
        st.subheader("Convert Data Types")
        all_cols = df.columns.tolist()
        selected_cols = st.multiselect("Select Columns to Convert", all_cols)
        new_type = st.selectbox("Select New Data Type", ["int64", "float64", "string", "datetime", "category"])
        
        if st.button("Convert Data Types"):
            for col in selected_cols:
                try:
                    if new_type == "datetime":
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    elif new_type == "category":
                        df[col] = df[col].astype('category')
                    else:
                        df[col] = df[col].astype(new_type)
                except Exception as e:
                    st.warning(f"Could not convert {col}: {str(e)}")
            
            st.session_state.df_processed = df.copy()
            st.success(f"✅ Converted {len(selected_cols)} columns to {new_type}")
            add_to_history("Type Conversion", f"Converted {selected_cols} to {new_type}")
        
        # Column Operations
        st.subheader("Column Operations")
        col_op = st.selectbox("Operation", ["Drop Columns", "Rename Columns", "Reorder Columns"])
        
        if col_op == "Drop Columns":
            cols_to_drop = st.multiselect("Select Columns to Drop", all_cols)
            if st.button("Drop Columns"):
                df = df.drop(columns=cols_to_drop)
                st.session_state.df_processed = df.copy()
                st.success(f"✅ Dropped {len(cols_to_drop)} columns")
                add_to_history("Drop Columns", f"Dropped {cols_to_drop}")
        
        elif col_op == "Rename Columns":
            col_to_rename = st.selectbox("Select Column to Rename", all_cols)
            new_name = st.text_input("New Column Name")
            if st.button("Rename Column"):
                df = df.rename(columns={col_to_rename: new_name})
                st.session_state.df_processed = df.copy()
                st.success(f"✅ Renamed {col_to_rename} to {new_name}")
                add_to_history("Rename Column", f"{col_to_rename} -> {new_name}")
        
        # Show processed data
        if st.session_state.df_processed is not None:
            st.subheader("Cleaned Data Preview")
            st.dataframe(st.session_state.df_processed.head(), use_container_width=True)
    else:
        st.warning("⚠️ Please upload a dataset first from the Home page.")

# Null Analysis Page
elif page == "🔢 Null Analysis":
    if st.session_state.df is not None:
        st.header("Null Value Analysis")
        df = st.session_state.df_processed if st.session_state.df_processed is not None else st.session_state.df
        
        # Null Summary
        null_counts = df.isnull().sum()
        null_percentages = (null_counts / len(df) * 100).round(2)
        
        null_df = pd.DataFrame({
            'Column': null_counts.index,
            'Null Count': null_counts.values,
            'Null Percentage': null_percentages.values
        })
        null_df = null_df[null_df['Null Count'] > 0].sort_values('Null Count', ascending=False)
        
        st.subheader("Null Value Summary")
        if len(null_df) > 0:
            st.dataframe(null_df, use_container_width=True)
            
            # Visualization
            fig = px.bar(null_df, x='Column', y='Null Count', 
                        title='Null Values by Column',
                        labels={'Null Count': 'Number of Null Values'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No null values found in the dataset!")
        
        # Handle Null Values
        st.subheader("Handle Null Values")
        cols_with_nulls = null_df['Column'].tolist() if len(null_df) > 0 else []
        
        if cols_with_nulls:
            selected_col = st.selectbox("Select Column to Handle Nulls", cols_with_nulls)
            method = st.selectbox("Handling Method", [
                "Drop Rows",
                "Fill with Mean (numeric)",
                "Fill with Median (numeric)",
                "Fill with Mode",
                "Fill with Forward Fill (ffill)",
                "Fill with Backward Fill (bfill)",
                "Fill with Custom Value"
            ])
            
            if method == "Fill with Custom Value":
                custom_value = st.text_input("Enter Custom Value")
            else:
                custom_value = None
            
            if st.button("Apply Null Handling"):
                initial_nulls = df[selected_col].isnull().sum()
                
                if method == "Drop Rows":
                    df = df.dropna(subset=[selected_col])
                elif method == "Fill with Mean (numeric)":
                    if df[selected_col].dtype in [np.number]:
                        df[selected_col].fillna(df[selected_col].mean(), inplace=True)
                    else:
                        st.warning("Column is not numeric!")
                elif method == "Fill with Median (numeric)":
                    if df[selected_col].dtype in [np.number]:
                        df[selected_col].fillna(df[selected_col].median(), inplace=True)
                    else:
                        st.warning("Column is not numeric!")
                elif method == "Fill with Mode":
                    df[selected_col].fillna(df[selected_col].mode()[0] if len(df[selected_col].mode()) > 0 else 0, inplace=True)
                elif method == "Fill with Forward Fill (ffill)":
                    df[selected_col] = df[selected_col].ffill()
                elif method == "Fill with Backward Fill (bfill)":
                    df[selected_col] = df[selected_col].bfill()
                elif method == "Fill with Custom Value" and custom_value:
                    df[selected_col].fillna(custom_value, inplace=True)
                
                st.session_state.df_processed = df.copy()
                st.success(f"✅ Handled {initial_nulls} null values in {selected_col}")
                add_to_history("Null Handling", f"{method} on {selected_col}")
            
            # Quick actions
            st.subheader("Quick Actions")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Fill All Numeric with Mean"):
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        df[col].fillna(df[col].mean(), inplace=True)
                    st.session_state.df_processed = df.copy()
                    st.success("✅ Filled all numeric columns with mean")
                    add_to_history("Null Handling", "Fill all numeric with mean")
            with col2:
                if st.button("Fill All Categorical with Mode"):
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                    for col in categorical_cols:
                        mode_val = df[col].mode()
                        if len(mode_val) > 0:
                            df[col].fillna(mode_val[0], inplace=True)
                    st.session_state.df_processed = df.copy()
                    st.success("✅ Filled all categorical columns with mode")
                    add_to_history("Null Handling", "Fill all categorical with mode")
        else:
            st.info("No columns with null values to handle.")
    else:
        st.warning("⚠️ Please upload a dataset first from the Home page.")

# Data Processing Page
elif page == "⚙️ Data Processing":
    if st.session_state.df is not None:
        st.header("Data Processing")
        df = st.session_state.df_processed if st.session_state.df_processed is not None else st.session_state.df
        
        st.subheader("Feature Engineering")
        
        # Create new column
        st.write("**Create New Column**")
        new_col_name = st.text_input("New Column Name")
        col1, col2 = st.columns(2)
        with col1:
            operation = st.selectbox("Operation", ["Addition", "Subtraction", "Multiplication", "Division", "Custom Expression"])
        with col2:
            if operation != "Custom Expression":
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(num_cols) >= 2:
                    col_a = st.selectbox("Column A", num_cols)
                    col_b = st.selectbox("Column B", num_cols)
        
        if operation == "Custom Expression":
            expression = st.text_input("Enter Expression (use df['col'] syntax)", 
                                     help="Example: df['price'] * df['quantity']")
            if st.button("Create Column") and new_col_name and expression:
                try:
                    df[new_col_name] = eval(expression)
                    st.session_state.df_processed = df.copy()
                    st.success(f"✅ Created column '{new_col_name}'")
                    add_to_history("Create Column", f"{new_col_name} = {expression}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        elif operation and new_col_name and col_a and col_b:
            if st.button("Create Column"):
                try:
                    if operation == "Addition":
                        df[new_col_name] = df[col_a] + df[col_b]
                    elif operation == "Subtraction":
                        df[new_col_name] = df[col_a] - df[col_b]
                    elif operation == "Multiplication":
                        df[new_col_name] = df[col_a] * df[col_b]
                    elif operation == "Division":
                        df[new_col_name] = df[col_a] / df[col_b].replace(0, np.nan)
                    st.session_state.df_processed = df.copy()
                    st.success(f"✅ Created column '{new_col_name}'")
                    add_to_history("Create Column", f"{new_col_name} = {col_a} {operation} {col_b}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Filtering
        st.subheader("Filter Data")
        filter_col = st.selectbox("Select Column to Filter", df.columns.tolist())
        
        if df[filter_col].dtype in [np.number]:
            min_val = st.number_input("Minimum Value", value=float(df[filter_col].min()))
            max_val = st.number_input("Maximum Value", value=float(df[filter_col].max()))
            if st.button("Apply Filter"):
                df = df[(df[filter_col] >= min_val) & (df[filter_col] <= max_val)]
                st.session_state.df_processed = df.copy()
                st.success(f"✅ Filter applied. New shape: {df.shape}")
                add_to_history("Filter", f"{filter_col} between {min_val} and {max_val}")
        else:
            unique_vals = df[filter_col].unique().tolist()
            selected_vals = st.multiselect("Select Values to Keep", unique_vals)
            if st.button("Apply Filter"):
                df = df[df[filter_col].isin(selected_vals)]
                st.session_state.df_processed = df.copy()
                st.success(f"✅ Filter applied. New shape: {df.shape}")
                add_to_history("Filter", f"{filter_col} in {selected_vals}")
        
        # Sorting
        st.subheader("Sort Data")
        sort_col = st.selectbox("Sort By Column", df.columns.tolist())
        ascending = st.checkbox("Ascending", value=True)
        if st.button("Sort Data"):
            df = df.sort_values(by=sort_col, ascending=ascending)
            st.session_state.df_processed = df.copy()
            st.success(f"✅ Data sorted by {sort_col}")
            add_to_history("Sort", f"Sorted by {sort_col}, ascending={ascending}")
        
        # Group By
        st.subheader("Group By & Aggregate")
        group_cols = st.multiselect("Select Columns to Group By", df.columns.tolist())
        agg_col = st.selectbox("Select Column to Aggregate", df.select_dtypes(include=[np.number]).columns.tolist() if len(df.select_dtypes(include=[np.number]).columns) > 0 else [])
        agg_func = st.selectbox("Aggregation Function", ["mean", "sum", "count", "min", "max", "median", "std"])
        
        if st.button("Apply Group By") and group_cols and agg_col:
            grouped = df.groupby(group_cols)[agg_col].agg(agg_func).reset_index()
            st.dataframe(grouped, use_container_width=True)
            if st.button("Replace Dataset with Grouped Data"):
                df = grouped
                st.session_state.df_processed = df.copy()
                st.success("✅ Dataset replaced with grouped data")
                add_to_history("Group By", f"Grouped by {group_cols}, {agg_func} of {agg_col}")
    else:
        st.warning("⚠️ Please upload a dataset first from the Home page.")

# Visualizations Page
elif page == "📈 Visualizations":
    if st.session_state.df is not None:
        st.header("Data Visualizations")
        df = st.session_state.df_processed if st.session_state.df_processed is not None else st.session_state.df
        
        st.subheader("Create Visualizations")
        viz_type = st.selectbox("Select Visualization Type", [
            "Bar Chart",
            "Line Chart",
            "Scatter Plot",
            "Histogram",
            "Box Plot",
            "Heatmap (Correlation)",
            "Pie Chart"
        ])
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if viz_type == "Bar Chart":
            x_col = st.selectbox("X-Axis (Categorical)", categorical_cols if categorical_cols else df.columns.tolist())
            y_col = st.selectbox("Y-Axis (Numeric)", numeric_cols if numeric_cols else df.columns.tolist())
            if x_col and y_col:
                fig = px.bar(df, x=x_col, y=y_col, title=f'{y_col} by {x_col}')
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Line Chart":
            x_col = st.selectbox("X-Axis", df.columns.tolist())
            y_col = st.selectbox("Y-Axis (Numeric)", numeric_cols)
            if x_col and y_col:
                fig = px.line(df, x=x_col, y=y_col, title=f'{y_col} over {x_col}')
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Scatter Plot":
            x_col = st.selectbox("X-Axis (Numeric)", numeric_cols)
            y_col = st.selectbox("Y-Axis (Numeric)", numeric_cols)
            color_col = st.selectbox("Color By (Optional)", [None] + df.columns.tolist())
            if x_col and y_col:
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f'{y_col} vs {x_col}')
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Histogram":
            col = st.selectbox("Select Column", numeric_cols)
            bins = st.slider("Number of Bins", 10, 100, 30)
            if col:
                fig = px.histogram(df, x=col, nbins=bins, title=f'Distribution of {col}')
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Box Plot":
            y_col = st.selectbox("Y-Axis (Numeric)", numeric_cols)
            x_col = st.selectbox("X-Axis (Optional, Categorical)", [None] + categorical_cols)
            if y_col:
                fig = px.box(df, x=x_col, y=y_col, title=f'Box Plot of {y_col}')
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Heatmap (Correlation)":
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix, 
                              labels=dict(x="Columns", y="Columns", color="Correlation"),
                              x=corr_matrix.columns,
                              y=corr_matrix.columns,
                              title="Correlation Heatmap",
                              aspect="auto")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 numeric columns for correlation")
        
        elif viz_type == "Pie Chart":
            col = st.selectbox("Select Column", categorical_cols if categorical_cols else df.columns.tolist())
            if col:
                value_counts = df[col].value_counts()
                fig = px.pie(values=value_counts.values, names=value_counts.index, title=f'Distribution of {col}')
                st.plotly_chart(fig, use_container_width=True)
        
        # Quick Stats
        st.subheader("Quick Statistics")
        if numeric_cols:
            selected_stat_col = st.selectbox("Select Column for Statistics", numeric_cols)
            if selected_stat_col:
                stats_df = pd.DataFrame({
                    'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', '25th Percentile', '75th Percentile'],
                    'Value': [
                        df[selected_stat_col].mean(),
                        df[selected_stat_col].median(),
                        df[selected_stat_col].std(),
                        df[selected_stat_col].min(),
                        df[selected_stat_col].max(),
                        df[selected_stat_col].quantile(0.25),
                        df[selected_stat_col].quantile(0.75)
                    ]
                })
                st.dataframe(stats_df, use_container_width=True)
    else:
        st.warning("⚠️ Please upload a dataset first from the Home page.")

# AI Assistant Page
elif page == "🤖 AI Assistant":
    st.header("AI Assistant (Ollama - Llama3.2)")
    
    # Check connection
    st.info("💡 Make sure Ollama is running with llama3.2 model installed. Run: `ollama run llama3.2`")
    
    if st.session_state.df is not None:
        df = st.session_state.df_processed if st.session_state.df_processed is not None else st.session_state.df
        
        # Context about the data
        data_context = f"""
        Dataset Info:
        - Shape: {df.shape[0]} rows, {df.shape[1]} columns
        - Columns: {', '.join(df.columns.tolist()[:10])}{'...' if len(df.columns) > 10 else ''}
        - Data types: {dict(df.dtypes)}
        - Null values: {df.isnull().sum().sum()}
        """
        
        st.subheader("Data Context")
        st.text(data_context)
        
        # Preset questions
        st.subheader("Quick Questions")
        quick_questions = [
            "What cleaning steps should I perform on this dataset?",
            "What visualizations would be useful for this data?",
            "What are potential issues with this dataset?",
            "Suggest feature engineering ideas for this data",
            "What statistical analysis should I perform?"
        ]
        
        selected_q = st.selectbox("Select a question or type your own", [""] + quick_questions)
        
        user_query = st.text_area("Your Question to AI", value=selected_q, height=150,
                                 placeholder="Ask anything about your data analysis...")
        
        if st.button("Ask AI Assistant"):
            if user_query:
                with st.spinner("🤔 Thinking..."):
                    prompt = f"""You are a data analysis assistant. Help the user with their data analysis question.

{data_context}

User Question: {user_query}

Provide clear, actionable advice:"""
                    response = call_ollama(prompt)
                    st.subheader("AI Response")
                    st.markdown(response)
            else:
                st.warning("Please enter a question")
        
        # Data-specific suggestions
        st.subheader("AI Suggestions")
        if st.button("Get Analysis Suggestions"):
            with st.spinner("🤔 Analyzing your data..."):
                prompt = f"""Based on this dataset:
{data_context}

Provide 5 specific data analysis recommendations. Be concise and actionable."""
                suggestions = call_ollama(prompt)
                st.markdown(suggestions)
    else:
        st.warning("⚠️ Please upload a dataset first from the Home page.")
        
        # General AI chat
        st.subheader("General AI Assistance")
        general_query = st.text_area("Ask a general data analysis question", height=100)
        if st.button("Ask"):
            if general_query:
                with st.spinner("🤔 Thinking..."):
                    response = call_ollama(f"You are a data analysis expert. Answer: {general_query}")
                    st.markdown(response)

# Export Data Page
elif page == "💾 Export Data":
    if st.session_state.df_processed is not None:
        st.header("Export Processed Data")
        
        df = st.session_state.df_processed
        
        st.subheader("Final Dataset Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        with col4:
            st.metric("Null Values", df.isnull().sum().sum())
        
        st.subheader("Preview Final Data")
        st.dataframe(df, use_container_width=True)
        
        st.subheader("Download Options")
        export_format = st.selectbox("Select Export Format", ["CSV", "Excel", "JSON"])
        
        if export_format == "CSV":
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        elif export_format == "Excel":
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Processed Data')
            excel_data = output.getvalue()
            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        elif export_format == "JSON":
            json_str = df.to_json(orient='records', indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str.encode('utf-8'),
                file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Processing History
        st.subheader("Processing History")
        if st.session_state.processing_history:
            history_df = pd.DataFrame(st.session_state.processing_history)
            st.dataframe(history_df, use_container_width=True)
        else:
            st.info("No processing history yet.")
        
        # Reset option
        st.subheader("Reset Dataset")
        if st.button("Reset to Original Data"):
            st.session_state.df_processed = st.session_state.df.copy()
            st.session_state.processing_history = []
            st.success("✅ Dataset reset to original")
            st.rerun()
    else:
        st.warning("⚠️ Please upload and process a dataset first.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>AI-Powered Data Analysis Suite | Powered by Streamlit & Ollama</div>", unsafe_allow_html=True)

