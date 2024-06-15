#oracle.py

import pandas as pd
import json
from config import LEAGUES, num_last_matches

# Define the column indexes for each sheet
column_indexes = {
    "league": "Div",
    "Date": "Date",
    "Time": "Time",
    "HomeTeam": "HomeTeam",
    "AwayTeam": "AwayTeam",
    "AvgH": "AvgH",
    "AvgD": "AvgD",
    "AvgA": "AvgA",
    "AvgMORE25": "Avg>2.5",
    "AvgCLESS25": "Avg<2.5"
}

# Function to parse the match row
def parse_match_row(row):
    match_data = {key: row[value] for key, value in column_indexes.items()}
    # Convert numerical values to float
    for key in ["AvgH", "AvgD", "AvgA", "AvgMORE25", "AvgCLESS25"]:
        match_data[key] = float(match_data[key])
    # Convert date and time to strings
    match_data["Date"] = match_data["Date"].strftime("%Y-%m-%d")
    match_data["Time"] = match_data["Time"].strftime("%H:%M")
    return match_data

# Function to generate future matches JSON
def generate_future_matches():
    # Read data from Excel file
    file_path = "static/data/2324/all-euro-data-2023-2024-last.xlsx"
    excel_data = pd.read_excel(file_path, sheet_name=None)

    # Debug: Print the names of the sheets in the Excel file
    print("Sheets in Excel file:", excel_data.keys())

    # Generate JSON output for all leagues
    all_leagues_data = {}

    for league_code, sheet_name in LEAGUES.items():
        if league_code in excel_data:
            print(f"Processing sheet: {sheet_name} (Code: {league_code})")
            df = excel_data[league_code]
            print(f"Number of rows in {sheet_name}: {len(df)}")
            # Select the last `num_last_matches[league_code]` rows
            last_x_rows = df.tail(num_last_matches.get(league_code, 10))  # Default to 10 if not specified
            json_output = [parse_match_row(row) for index, row in last_x_rows.iterrows()]
            all_leagues_data[league_code] = json_output
        else:
            print(f"Sheet {league_code} not found in Excel file")

    # Convert to JSON string
    json_string = json.dumps(all_leagues_data, indent=4)

    # Save JSON string to a file
    output_file = 'static/future_matches.json'
    with open(output_file, 'w') as f:
        f.write(json_string)

    print(f"JSON output saved to {output_file}")

# Execute directly
if __name__ == '__main__':
    generate_future_matches()
