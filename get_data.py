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
start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 1, 2)

all_data = []
current_date = start_date

while current_date <= end_date:
    # Calculate end of day (or end date if sooner)
    week_end = current_date
    week_end = min(week_end, end_date)
    
    print(f"Fetching data from {current_date.date()} to {week_end.date()}...")
    
    # Fetch data for this week
    df = client.get_dataset(
        dataset="isone_lmp_day_ahead_hourly",
        start=current_date.strftime("%Y-%m-%d"),
        end=week_end.strftime("%Y-%m-%d"),
    )
    
    all_data.append(df)
    
    # Move to next week
    current_date = week_end + timedelta(days=0)
    
    # Delay between requests (account for multiple internal pages per request)
    # API limit is 1 request per second, but each get_dataset() may make multiple requests
    if current_date <= end_date:
        time.sleep(10)  # Wait 10 seconds between requests for daily chunks

# Combine all data
df = pd.concat(all_data, ignore_index=True)

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Save to CSV
csv_path = "data/isone_lmp_data.csv"
df.to_csv(csv_path, index=False)

print(f"\nData saved to {csv_path}")
print(f"Total rows: {len(df)}")
print(f"\nFirst few rows:")
print(df.head())
