# Account Outlier Detection - User Guide

## Quick Start

### First Time Setup

1. **Extract Files**
   - Unzip the entire folder to your desired location
   - Do NOT move individual files

2. **Run Setup** (Only needed once)
   - Double-click `setup.bat`
   - Wait for installation to complete (2-5 minutes)
   - This creates a virtual environment and installs dependencies

3. **Launch App**
   - Double-click `launch_app.bat`
   - Your browser will open automatically
   - If not, go to: http://localhost:8501

### Using the App

#### Step 1: Upload Data
- Click "Browse files" or drag and drop
- Supported formats: CSV, Excel (.xlsx, .xls)
- Review data preview and column information

#### Step 2: Configure Analysis
1. **Set Filters**
   - Choose which data to include
   - Apply filters and verify row counts

2. **Configure Model**
   - Select dependent variable (what you're predicting)
   - Choose continuous variables (will be log-transformed)
   - Choose categorical variables (will be one-hot encoded)

3. **Set K-Fold Parameters**
   - Click "Find Optimal K" for automatic suggestion
   - Or manually set number of folds
   - Choose time period and account columns

4. **Set Outlier Detection**
   - Adjust contamination rate (expected % of outliers)
   - Fine-tune DBSCAN/HDBSCAN parameters if needed

5. **Save Configuration** (Optional)
   - Save your settings to reuse later
   - Load from sidebar

#### Step 3: Run Analysis
- Click "Run Complete Analysis"
- Wait for progress bar to complete (1-5 minutes depending on data size)
- View summary metrics

#### Step 4: View Results
- **Overview**: See model performance and K-Fold results
- **Outliers**: List of flagged accounts with statistics
- **3D Plot**: Interactive visualization (change axes with dropdowns)
- **Export**: Download results as CSV

#### Step 5: Account Charts
- View time series for all accounts or outliers only
- Charts show units, base units, prices over time
- Consistent date ranges for easy comparison

### Tips

- **Performance**: Limit account charts to 20-30 at a time
- **Configurations**: Save frequently used settings
- **Export**: Download results before closing app
- **Multiple Runs**: Compare different filter/model combinations

### Troubleshooting

**App won't start:**
- Run `setup.bat` again
- Check that Python 3.11 is installed
- Ensure no antivirus blocking

**Browser doesn't open:**
- Manually go to: http://localhost:8501
- Check firewall settings

**Out of memory:**
- Reduce data size with filters
- Limit account charts displayed
- Close other applications

**Charts not showing:**
- Check that data has Date column
- Ensure Date format is recognized

### System Requirements

- **OS**: Windows 10 or later
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB free space
- **Browser**: Chrome, Firefox, or Edge

### Support

For issues or questions:
1. Check this guide first
2. Review error messages in console
3. Contact your IT/Analytics team

---

**Version 1.0 | Last Updated: 2025**
