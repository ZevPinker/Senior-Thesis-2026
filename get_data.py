import os
import sys
import time
from datetime import datetime, timedelta

from dotenv import load_dotenv
from gridstatusio import GridStatusClient
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
GRIDSTATUS_API_KEY = os.getenv("GRIDSTATUS_API_KEY")
if not GRIDSTATUS_API_KEY:
    raise ValueError(
        "GRIDSTATUS_API_KEY not found in environment. "
        "Please set it in your .env file."
    )

client = GridStatusClient(GRIDSTATUS_API_KEY)

# Fetch data in weekly chunks with delays
start_date = datetime(year=2025, month=1, day=2)  # January 2 (market open after New Year)
end_date = datetime(year=2025, month=12, day=31)   # Through end of year

all_data = []
current_date = start_date

while current_date < end_date:
    # Calculate end of week (or end date if sooner)
    week_end = current_date + timedelta(days=7)
    week_end = min(week_end, end_date)
    
    print(f"Fetching data from {current_date.date()} to {week_end.date()}...")
    
    # Fetch data for this week
    df = client.get_dataset(
        dataset="isone_lmp_real_time_hourly_final",
        start=current_date.strftime("%Y-%m-%d"),
        end=week_end.strftime("%Y-%m-%d"),
        filter_column="location",
        filter_value=[
            "DR.CT_Eastern",
            "DR.CT_Northern",
            "DR.CT_Norwalk-Stamford",
            "DR.CT_Western",
            "DR.CT_Western_SWCT"
        ],
        filter_operator="in"
    )
    print(f"Retrieved {len(df)} rows")
    if len(df) > 0:
        print(df.head())
    all_data.append(df)
    
    # Move to next week
    current_date = week_end + timedelta(days=1)
    
    # Delay between requests (API limit is 1 request per second)
    if current_date < end_date:
        print("Waiting 2 seconds before next request...")
        time.sleep(2)  # Wait 2 seconds between requests

# Combine all data
if all_data:
    df = pd.concat(all_data, ignore_index=True)
else:
    print("ERROR: No data was retrieved!")
    df = pd.DataFrame()

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Save to CSV
csv_path = "data/isone_real_time_final_lmp_data.csv"
df.to_csv(csv_path, index=False)

print(f"\nData saved to {csv_path}")
print(f"Total rows: {len(df)}")
print(f"\nFirst few rows:")
print(df.head())
