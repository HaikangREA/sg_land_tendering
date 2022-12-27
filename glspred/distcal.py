# to calculate pairwise distance between poi (for land parcels, can also calculate the time difference in launch)
# input: poi table a, poi table b (both with cols [id, latitude, longitude])
# output: df[id_a, id_b, distance]

import pandas as pd
import numpy as np
from geopy.distance import geodesic
from tqdm import tqdm
from datetime import date, datetime


class PairwiseDistCalculator:

    def __init__(self, poi_master=None, poi_ref=None, master_id=None, ref_id=None):
        self.poi_master = poi_master
        self.poi_reference = poi_ref
        self.master_id = master_id
        self.reference_id = ref_id

    def calculate_distance(self, poi_master=None, poi_ref=None, master_id=None, ref_id=None, distance_limit=5000):
        if poi_master is not None:
            self.poi_master = poi_master
        if poi_ref is not None:
            self.poi_reference = poi_ref
        if master_id is not None:
            self.master_id = master_id
        if ref_id is not None:
            self.reference_id = ref_id

        key_a = self.master_id
        key_b = self.reference_id

        loop_a, loop_b = self.poi_master[key_a], self.poi_reference[key_b]
        self.poi_master.index = loop_a
        self.poi_reference.index = loop_b
        output = pd.DataFrame(columns=['poi_a',
                                       'poi_b',
                                       'distance'])

        for id_a in tqdm(loop_a, desc='Calculating pairwise distance'):
            coord_a = self.poi_master.loc[id_a].coordinates
            for id_b in loop_b:
                coord_b = (self.poi_reference.loc[id_b].latitude, self.poi_reference.loc[id_b].longitude)
                try:
                    distance = round(geodesic(coord_a, coord_b).m, 2)
                except ValueError:
                    distance = -1
                to_append = [id_a, id_b, distance]
                if 0 <= distance <= distance_limit:
                    output.loc[len(output)] = to_append

        return output

    @staticmethod
    def find_nearby(pairwise_tbl: pd.DataFrame, rename_col=None):
        header = pairwise_tbl.columns
        nearby_df = pairwise_tbl.groupby(header[0]).min(header[-1])
        if rename_col is not None:
            nearby_df = nearby_df.rename(columns={'distance': rename_col})
        return nearby_df

