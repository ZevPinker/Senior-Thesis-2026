# Senior Thesis 2026

A repository for the development of algorithms and experiments towards the completion of my senior thesis requirement in the Computer Science Major.

## Project Overview
This repository includes the implementation and analysis of an ADMM distributed optimization algorithm for BESS for virtual power plants. 

This project analyzes ISO New England (ISONE) Locational Marginal Pricing (LMP) data for real-time and day-ahead electricity markets. It includes data fetching utilities, algorithmic optimization models, and comprehensive visualization and analysis tools.

## Prerequisites

- **Python 3.12+** (recommended Python 3.12.0)
- **pip** (comes with Python)
- A GridStatus API key (free tier available)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Senior-Thesis-2026.git
cd Senior-Thesis-2026
```

### 2. Set Up Python Environment

#### Option A: Using Homebrew (macOS)

```bash
# Install Python 3.12
brew install python@3.12

# Create a virtual environment
python3.12 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

#### Option B: Using pyenv (Recommended for multiple Python versions)

```bash
# Install pyenv
brew install pyenv

# Install Python 3.12
pyenv install 3.12.0
pyenv local 3.12.0

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure API Key

1. **Create a GridStatus API account:**
   - Visit [https://gridstatus.io/](https://gridstatus.io/)
   - Sign up for a free account
   - Generate an API key from your dashboard

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   ```
   
3. **Edit the `.env` file:**
   ```bash
   # Open .env and replace with your actual API key
   GRIDSTATUS_API_KEY=your_api_key_here
   ```

## Project Structure

```
Senior-Thesis-2026/
├── get_data.py                 # Fetch ISONE LMP data from GridStatus API
├── requirements.txt            # Python package dependencies
├── .env.example               # Template for environment variables
├── README.md                  # This file
├── algos/                     # Optimization algorithms
│   ├── naive_coupled.py      # Baseline algorithm implementation
│   └── outputs/              # Algorithm outputs and results
├── data/                      # Data storage
│   └── isone_lmp_data.csv    # Downloaded LMP data (generated)
└── visualizations/            # Visualization scripts and outputs
    ├── vizualize_lmp.py      # LMP visualization generation
    └── fig*.png              # Generated visualization PNGs
```

## Quick Start

### Fetch Data

```bash
# Activate the virtual environment
source venv/bin/activate

# Download ISONE LMP data (Jan 2025)
python get_data.py
```

This will:
- Fetch day-ahead and real-time LMP prices from the GridStatus API
- Save results to `data/isone_lmp_data.csv`
- Display summary statistics

### Generate Visualizations

```bash
# Activate the virtual environment
source venv/bin/activate

# Generate publication-quality price analysis figures
python visualizations/vizualize_lmp.py
```

Generated visualizations are saved to the `visualizations/` folder:
- `fig1_timeseries.png` - Full year price time series
- `fig2_spread.png` - RT vs DA price spread analysis
- `fig3_daily_profile.png` - Average hourly price patterns
- `fig4_duration_curve.png` - Price duration curves
- `fig5a_heatmap_rt.png` & `fig5b_heatmap_da.png` - Monthly heatmaps
- `fig6_monthly_avg.png` - Monthly average prices
- `fig7_locations.png` - Multi-location comparison

### Run Algorithms

```bash
# Activate the virtual environment
source venv/bin/activate

# Run the naive coupling algorithm
python algos/naive_coupled.py
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `python-dotenv` | Latest | Environment variable management |
| `gridstatusio` | 0.15.1+ | GridStatus API client |
| `pandas` | 3.0+ | Data manipulation and analysis |
| `numpy` | 1.26+ | Numerical computing |
| `matplotlib` | 3.10+ | Data visualization |
| `cvxpy` | Latest | Convex optimization |

See `requirements.txt` for exact versions and all transitive dependencies.

## Troubleshooting

### API Key Issues

If you get `GRIDSTATUS_API_KEY not found in environment`:
1. Verify `.env` file exists in the project root
2. Check that your API key is correctly set: `GRIDSTATUS_API_KEY=your_key`
3. Restart your terminal session after creating `.env`

### Virtual Environment Issues

If packages aren't installing or import errors occur:
```bash
# Deactivate and remove old venv
deactivate
rm -rf venv

# Create a fresh environment
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Python Version Issues

Check your Python version:
```bash
python --version
python3.12 --version
```

For macOS with multiple Python versions, ensure you're using the correct one:
```bash
which python3.12
/opt/homebrew/bin/python3.12 -m venv venv
```

## License

See [LICENSE](LICENSE) for details.

## Notes for Developers

- Always activate the virtual environment before running scripts: `source venv/bin/activate`
- The `.env` file should never be committed to version control (see `.gitignore`)
- Data files in `data/` are generated and can be regenerated by running `get_data.py`
- All visualizations are high-resolution (200 DPI) suitable for thesis publication
