import os
import logging
from tcia_utils import nbia

# 1. Configuration (Optional but Recommended)
# Set logging to see the progress of the download
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

# Define the dataset collection name
collection_name = "LIDC-IDRI"

# Define the target directory for the download
# This will create a folder named 'LIDC-IDRI_Downloads' (or similar)
download_path = os.path.join(os.getcwd(), f"{collection_name}_Downloads")

print(f"Starting download for collection: {collection_name}")
print(f"Files will be saved to: {download_path}")

# 2. Get Series UIDs for the Collection
# The getSeries() function retrieves metadata (including UIDs) for all scans
# in the specified collection.
print("Querying for all Series Instance UIDs...")
series_uids_df = nbia.getSeries(
    collection=collection_name,
    # Use 'df' format to get results as a Pandas DataFrame
    format='df'
)

# Check if any UIDs were found
if series_uids_df.empty:
    print(f"No series UIDs found for collection {collection_name}. Check the name and API status.")
else:
    print(f"Found {len(series_uids_df)} series to download.")

    # 3. Download the Data
    # The downloadSeries() function takes the DataFrame of UIDs and downloads
    # all associated DICOM files and annotation XMLs.
    print("Starting data download (this may take a very long time for LIDC-IDRI)...")

    nbia.downloadSeries(
        series_uids_df,
        input_type='df',  # Specify that the input is a DataFrame
        path=download_path, # Specify the download directory
        # as_zip=False, # (Optional) Set to True to save as zip files instead of unzipped folders
        # csv_filename='lidc_download_log.csv', # (Optional) Save a log of downloaded series
        # number=10 # (Optional) Uncomment to test with only the first 10 series
    )

    print("Download process completed.")
