import pandas as pd

df = pd.read_csv('Oil_Wells_Data_Analysis/well_info_enriched.csv')

# Drop unenriched rows
df = df[df['de_found'] != 0]

# Select columns
columns = ['well_name', 'de_operator', 'county', 'state', 'address',
           'de_status', 'de_well_type', 'de_direction', 'de_closest_city',
           'de_latitude', 'de_longitude', 'de_oil_bbls', 'de_gas_mcf',
           'de_production_start', 'de_production_end']

df = df[columns]

df.to_csv('Oil_Wells_Data_Analysis/well-map/wells.csv')