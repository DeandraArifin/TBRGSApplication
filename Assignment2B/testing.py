import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

scats_data = pd.read_excel('Resources/Scats_Data_October_2006.xls', sheet_name='Data', skiprows=1)
scats_sites = scats_data[['SCATS Number', 'NB_LATITUDE', 'NB_LONGITUDE']].drop_duplicates()

#usecols range 5 because the valid columns that need to be parsed are the first 5, so it removes the trailing semicolons
scats_sites_data = pd.read_csv('Resources/SCATSSite.csv', sep=';', usecols=range(5), skiprows=9, on_bad_lines='skip', header=0)
#strip trailing whitespaces from column names
scats_sites_data.columns = scats_sites_data.columns.str.strip()



print(scats_data["SCATS Number"].unique())
print(scats_sites_data.columns)
G = nx.Graph()

for idx, row in scats_sites.iterrows():
    scats_id = row['SCATS Number']
    lat = row['NB_LATITUDE']
    lon = row['NB_LONGITUDE']
    G.add_node(scats_id, pos=(lon, lat))  

pos = nx.get_node_attributes(G, 'pos')

plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=True, node_size=50)
plt.title('SCATS Sites Locations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
