# PBMZI Stocks Dashboard - Development Plan

## Project Overview
Interactive Streamlit dashboard for PBMZI stocks analysis with two main pages:
1. **PBMZI Stocks Overview** - EDA visualizations
2. **Interaction of PBMZI Stocks** - MST network analysis

## Files Created

### 1. app.py (Main Application)
- **Purpose**: Main Streamlit application with complete dashboard functionality
- **Features**:
  - Two-page navigation system
  - Interactive sidebar filters (years and companies)
  - Professional styling with custom CSS
  - Comprehensive error handling and data validation

### 2. requirements.txt
- **Purpose**: Python dependencies for the project
- **Dependencies**: streamlit, pandas, numpy, plotly, networkx, pyvis, openpyxl, xlrd

## Page 1: PBMZI Stocks Overview
**Layout**: Two-column design to avoid scrolling

**Left Column**:
- Stock prices trend (line chart)
- 30-day rolling volatility (line chart)  
- Correlation matrix (heatmap)

**Right Column**:
- Logarithmic returns trend (line chart)
- 60-day rolling volatility (line chart)
- Maximum drawdown vs maximum return (scatter plot)

**Interactive Features**:
- Year and company filters in sidebar
- Hover tooltips with detailed information
- Professional color schemes and legends

## Page 2: Interaction of PBMZI Stocks
**Main Features**:
- Large MST network visualization for all years (2018-2023)
- Additional year-specific MST networks when filters applied
- Interactive node highlighting and hover information
- Network statistics display
- Edge weights table

**Interactive Features**:
- Clickable nodes with connection details
- Node size based on number of connections
- Color-coded nodes by degree centrality
- Hover tooltips showing distances and connections

## Key Technical Implementations

### Data Processing Functions:
- `load_data()`: Loads and validates Excel data
- `calculate_log_returns()`: Computes logarithmic returns
- `calculate_rolling_volatility()`: Rolling volatility calculations
- `max_drawdown()` & `max_return()`: Risk/return metrics
- `create_mst_network()`: MST creation with NetworkX

### Visualization Features:
- Plotly interactive charts with professional styling
- Custom CSS for enhanced UI appearance
- Responsive layout with proper column arrangements
- Error handling for edge cases (insufficient data, etc.)

### Network Analysis:
- Correlation-based distance matrix calculation
- Kruskal's algorithm for MST construction
- Interactive network visualization with Plotly
- Year-specific MST comparisons

## Success Criteria
✅ Professional dashboard layout without scrolling
✅ Interactive filters for years and companies
✅ All 6 required visualizations on page 1
✅ MST network with clickable nodes on page 2
✅ Additional year-specific MSTs when filtered
✅ Comprehensive error handling and data validation
✅ Modern, professional styling and user experience

## Usage Instructions
1. Place `cleaned_PBMZI.xlsx` in the project directory
2. Run: `streamlit run app.py`
3. Use sidebar filters to customize analysis
4. Navigate between pages using the selectbox
5. Interact with charts and network visualizations
