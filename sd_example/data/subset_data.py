from __future__ import print_function
import pandas as pd
import pandana as pdna

# This script was used to create a subset of the San Diego data files

# Small section of San Diego
bbox = (-117.157516, 32.715666, -117.095032, 32.767068)
west, south, east, north = bbox

new_store = pd.HDFStore('sandag_subset.h5', 'w')
new_net_store = pd.HDFStore('osm_sandag_subset.h5', 'w')

with pd.HDFStore('sandag.h5') as store:
    with pd.HDFStore('osm_sandag.h5') as net_store:

        print('Read network')
        nodes, edges = net_store.nodes, net_store.edges
        net = pdna.Network(nodes["x"], nodes["y"], edges["from"], edges["to"],
                           edges[["weight"]])
        net.precompute(3000)

        print('Assign parcels to nodes')
        parcels = store.parcels
        parcels['node_id'] = net.get_node_ids(parcels['x'], parcels['y'])

        print('Subset nodes, then edges, to bounding box')
        new_nodes = nodes.loc[(nodes.x > west)
                              & (nodes.x < east)
                              & (nodes.y > south)
                              & (nodes.y < north)]

        new_edges = edges.loc[(edges['from'].isin(new_nodes.index))
                              & (edges['to'].isin(new_nodes.index))]

        print('Save to new network store')
        new_net_store.put('edges', new_edges)
        new_net_store.put('nodes', new_nodes)

        print('Save other tables to new store')

        print('parcels')
        new_parcels = parcels.loc[
            parcels.node_id.isin(new_nodes.index)].drop(
            ['apn', 'block_geoid'], axis=1)
        new_store.put('parcels', new_parcels)
        zones = new_parcels.taz_id.unique().tolist()

        print('buildings')
        new_buildings = store.buildings.loc[
            store.buildings.parcel_id.isin(new_parcels.index)].drop(
            'note', axis=1)
        new_store.put('buildings', new_buildings)

        print('households')
        new_households = store.households.loc[
            (store.households.building_id.isin(new_buildings.index)) |
            (store.households.building_id == -1)
        ]
        new_store.put('households', new_households)

        print('jobs')
        new_jobs = store.jobs.loc[
            (store.jobs.building_id.isin(new_buildings.index)) |
            (store.jobs.building_id == -1)
        ]
        new_store.put('jobs', new_jobs)

        other_tables = ['annual_household_control_totals',
                        'assessor_transactions',
                        'building_sqft_per_job',
                        'costar',
                        'fee_schedule',
                        'hh_controls',
                        'parcel_fee_schedule',
                        'scheduled_development_events',
                        'zoning',
                        'zoning_allowed_uses']

        for table in other_tables:
            print('Adding {} to store'.format(table))
            t = store[table]
            if 'name' in t.columns:
                t['name'] = t['name'].astype(str)
            if 'note' in t.columns:
                t['note'] = t['note'].astype(str)
            t.index.name = str(t.index.name)
            t.columns = [str(name) for name in t.columns]
            new_store.put(table, store[table])

        travel_data = store.travel_data
        travel_data.sort_index(inplace=True)
        travel_data = travel_data.loc[(zones, zones), :]
        travel_data.index.names = [
            str(name) for name in store.travel_data.index.names]
        travel_data.columns = [
            str(name) for name in store.travel_data.columns]

        new_store.put('travel_data', travel_data)

new_store.close()
new_net_store.close()
