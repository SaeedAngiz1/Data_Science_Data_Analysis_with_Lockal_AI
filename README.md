# AI-Powered Data Analysis Suite

https://data-science-ai.streamlit.app

A comprehensive Streamlit application for data cleaning, processing, visualization, and analysis with AI assistance from Ollama's local LLM (llama3.2).

## Features

### 📊 Data Management
- **File Upload**: Support for CSV, Excel, and JSON files
- **Data Overview**: Complete dataset information, statistics, and column details
- **Data Preview**: Interactive data exploration

### 🧹 Data Cleaning
- **Remove Duplicates**: Eliminate duplicate rows
- **Outlier Handling**: Detect and handle outliers using IQR or Z-Score methods
- **Data Type Conversion**: Convert columns to appropriate data types
- **Column Operations**: Drop, rename, or reorder columns

### 🔢 Null Value Analysis
- **Null Detection**: Visualize null values across all columns
- **Null Handling**: Multiple strategies including:
  - Drop rows with nulls
  - Fill with mean/median/mode
  - Forward fill / Backward fill
  - Custom value filling
- **Quick Actions**: Bulk operations for numeric and categorical columns

### ⚙️ Data Processing
- **Feature Engineering**: Create new columns with mathematical operations
- **Data Filtering**: Filter data by numeric or categorical values
- **Sorting**: Sort data by any column
- **Group By & Aggregation**: Group data and apply aggregations (mean, sum, count, etc.)

### 📈 Visualizations
- **Bar Charts**: Compare values across categories
- **Line Charts**: Track trends over time
- **Scatter Plots**: Identify relationships between variables
- **Histograms**: Understand data distributions
- **Box Plots**: Detect outliers and distributions
- **Heatmaps**: Correlation analysis
- **Pie Charts**: Categorical distributions

### 🤖 AI Assistant (Ollama Integration)
- **Context-Aware Analysis**: AI understands your dataset
- **Smart Suggestions**: Get recommendations for data analysis
- **Interactive Q&A**: Ask questions about your data
- **Quick Questions**: Preset questions for common tasks

### 💾 Export Options
- **Multiple Formats**: Export to CSV, Excel, or JSON
- **Processing History**: Track all changes made to your data
- **Reset Functionality**: Revert to original dataset

## Installation

### Prerequisites
- Python 3.8 or higher
- Ollama installed and running locally
- Llama3.2 model installed in Ollama

### Step 1: Install Ollama

Visit [ollama.ai](https://ollama.ai) and install Ollama for your operating system.

### Step 2: Install Llama3.2 Model

```bash
ollama pull llama3.2
```

### Step 3: Verify Ollama is Running

Start Ollama (it should run automatically after installation), then test:

```bash
ollama run llama3.2
```

### Step 4: Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Start the Streamlit App

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

### Workflow

1. **Upload Data**: Go to "Home & Upload" page and upload your dataset
2. **Explore Data**: Check "Data Overview" to understand your dataset
3. **Clean Data**: Use "Data Cleaning" to remove duplicates, handle outliers, etc.
4. **Handle Nulls**: Analyze and fix missing values in "Null Analysis"
5. **Process Data**: Create features and transform data in "Data Processing"
6. **Visualize**: Create charts and graphs in "Visualizations"
7. **AI Assistance**: Get help from the AI assistant throughout the process
8. **Export**: Download your processed data in "Export Data"

## AI Assistant Setup

The AI assistant uses Ollama's local LLM. Make sure:

1. Ollama is running (check with `ollama list`)
2. Llama3.2 model is installed (`ollama pull llama3.2`)
3. Default API is accessible at `http://localhost:11434`

If your Ollama is running on a different port, modify the `OLLAMA_API_URL` in `app.py`:

```python
OLLAMA_API_URL = "http://localhost:YOUR_PORT/api/generate"
```

## Features in Detail

### Data Cleaning Capabilities
- Automatic duplicate detection and removal
- Statistical outlier detection (IQR, Z-Score)
- Flexible data type conversion
- Batch column operations

### Null Value Handling
- Visual null value analysis
- Multiple imputation strategies
- Column-specific or bulk operations
- Smart defaults (mean for numeric, mode for categorical)

### Advanced Processing
- Custom expression evaluation for feature engineering
- Multi-column filtering
- Complex aggregations
- Data transformation pipelines

### Interactive Visualizations
- All charts are interactive (zoom, pan, hover)
- Export charts as images
- Multiple chart types for different data types
- Correlation analysis tools

## Troubleshooting

### Ollama Connection Issues
- Ensure Ollama is running: `ollama list`
- Check if the API is accessible: `curl http://localhost:11434/api/tags`
- Verify llama3.2 is installed: `ollama list`

### File Upload Issues
- Check file format (CSV, Excel, JSON supported)
- Ensure file is not corrupted
- Try smaller files first for testing

### Performance Issues
- Large datasets (>100k rows) may be slower
- Consider sampling for initial exploration
- Use filtering to reduce dataset size

## Example Use Cases

1. **Sales Data Analysis**
   - Clean transaction records
   - Handle missing customer information
   - Create revenue metrics
   - Visualize sales trends

2. **Customer Data Processing**
   - Remove duplicate customers
   - Fill missing demographics
   - Create customer segments
   - Analyze customer behavior

3. **Research Data**
   - Clean experimental data
   - Handle outliers appropriately
   - Create derived variables
   - Statistical visualizations

## Tips

- Use the AI Assistant for guidance on which cleaning steps to perform
- Check processing history to track changes
- Export intermediate results as backups
- Use visualizations to validate cleaning operations

## License

This project is open source and available for personal and commercial use.

## Support

For issues or questions:
- Check Ollama documentation: https://github.com/ollama/ollama
- Streamlit documentation: https://docs.streamlit.io
- Python pandas documentation: https://pandas.pydata.org/docs/

---

**Made with ❤️ using Streamlit, Ollama, and Python**

