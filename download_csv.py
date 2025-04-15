import requests

url = "http://data.insideairbnb.com/united-states/ny/new-york-city/2024-01-08/data/listings.csv.gz"
output_path = "listings.csv.gz"

# Downloading my file
response = requests.get(url)
with open(output_path, "wb") as f:
    f.write(response.content)
print("Starting download_csv.py script...")
