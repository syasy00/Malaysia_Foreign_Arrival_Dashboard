# Malaysia Foreign Arrival Dashboard 🇲🇾📊

[![Python](https://img.shields.io/badge/Python-100%25-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)](https://plotly.com)

> An interactive web dashboard that visualizes foreign arrival trends in Malaysia (2020–2023) using data visualization and machine learning forecasting. Built with Streamlit and Python for data-driven insights into tourism patterns.

## 📋 Overview

This project provides an interactive dashboard for analyzing Malaysia's foreign visitor arrival data from 2020 to 2023. The application uses advanced data visualization techniques and machine learning models to present historical trends and forecast future arrival patterns, helping stakeholders make informed decisions about tourism planning.

### Key Features

- 📊 **Interactive Visualizations**: Dynamic charts and graphs using Plotly
- 🔎 **Trend Analysis**: Historical data analysis of foreign arrivals
- 🤖 **Machine Learning Forecasting**: Predictive analytics for future trends
- 🌍 **Country-wise Breakdown**: Detailed analysis by country of origin
- 📅 **Time Series Analysis**: Monthly and yearly trend exploration
- 📱 **Responsive Design**: Works seamlessly on desktop and mobile
- 📄 **Data Export**: Download filtered data and visualizations

## 🛠️ Technologies Used

- **Frontend/Backend**: Streamlit
- **Language**: Python 3.8+
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib
- **Machine Learning**: Scikit-learn (for forecasting models)
- **Data Source**: Official Malaysia tourism statistics

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/syasy00/Malaysia_Foreign_Arrival_Dashboard.git
   cd Malaysia_Foreign_Arrival_Dashboard
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

   Required packages include:
   - streamlit
   - pandas
   - plotly
   - scikit-learn
   - numpy

3. **Run the application**
   ```bash
   streamlit run Foreign_Arrivals_Dashboard.py
   ```

4. **Open your browser**
   
   The dashboard will automatically open at `http://localhost:8501`

## 🎯 Usage

### Dashboard Navigation

1. **Home Page**
   - Overview of total arrivals
   - Key statistics and metrics
   - Quick insights

2. **Trend Analysis**
   - View historical arrival trends
   - Filter by date range
   - Compare different time periods

3. **Country Analysis**
   - Breakdown by country of origin
   - Top source countries
   - Geographic distribution

4. **Forecasting**
   - View ML-powered predictions
   - Adjust forecast parameters
   - Confidence intervals

### Interactive Features

- **Date Range Selector**: Filter data by specific periods
- **Country Filter**: Focus on specific source countries
- **Chart Type Selector**: Choose between different visualization styles
- **Download Options**: Export data and charts

## 📊 Data Insights

The dashboard reveals:
- **COVID-19 Impact**: Significant drop in 2020-2021
- **Recovery Trends**: Gradual recovery in 2022-2023
- **Top Source Markets**: Singapore, Indonesia, Thailand, China
- **Seasonal Patterns**: Peak periods and off-seasons
- **Growth Projections**: Future arrival forecasts

## 🏗️ Project Structure

```
Malaysia_Foreign_Arrival_Dashboard/
├── Foreign_Arrivals_Dashboard.py  # Main Streamlit app
├── final_cleaned_dataset.csv      # Cleaned data
├── countries_codes_and_coordinates.csv  # Country metadata
├── requirements.txt               # Python dependencies
├── runtime.txt                    # Python version
├── profile.png                    # Profile image
└── README.md                      # This file
```

## 💻 Technical Details

### Data Processing

- Data cleaning and preprocessing with Pandas
- Handling missing values and outliers
- Time series data transformation
- Feature engineering for ML models

### Visualizations

- Line charts for temporal trends
- Bar charts for country comparisons
- Heatmaps for seasonal patterns
- Geographic maps for distribution
- Interactive hover tooltips

### Machine Learning

- Time series forecasting models
- Trend analysis algorithms
- Seasonal decomposition
- Prediction confidence intervals

## 🚀 Deployment

This application can be deployed on:
- **Streamlit Cloud** (Recommended)
- **Heroku**
- **AWS/Azure**
- **Local server**

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy!

## 📚 Data Sources

- Malaysia Tourism Statistics
- Department of Statistics Malaysia
- Tourism Malaysia official data

## 🤝 Contributing

This is an academic project for data visualization and analysis coursework. Suggestions and improvements are welcome!

## 📄 License

This project is part of academic coursework at Universiti Utara Malaysia (UUM).

## 👨‍💻 Author

**Syasya** - [@syasy00](https://github.com/syasy00)

## 🙏 Acknowledgments

- Course: Data Visualization / Data Analytics
- Institution: Universiti Utara Malaysia (UUM)
- Data sources: Malaysia tourism authorities
- Streamlit community for excellent documentation

## 📧 Contact

For questions about the dashboard or data analysis methodology, feel free to reach out!

---

⭐ If you find this dashboard useful for understanding tourism trends, please give it a star!
