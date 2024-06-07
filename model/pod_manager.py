from typing import List

from model.pod import Pod
from engine import NetLogoCoordinate


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import manhattan_distances


class PodManager:
    def __init__(self):
        self.pods: List[Pod] = []
        self.id_to_pod = {}
        self.sku_to_pods = {}
        self.coordinate_to_pods = {}

    def add_pod(self, pod: Pod):
        self.pods.append(pod)
        self.coordinate_to_pods[(pod.pos_x, pod.pos_y)] = pod
        self.id_to_pod[pod.pod_id] = pod

        for sku in pod.skus:
            if sku not in self.sku_to_pods:
                self.sku_to_pods[sku] = []
            self.sku_to_pods[sku].append(pod)

    def add_sku_to_pod(self, sku: int, pod: Pod):
        if sku not in self.sku_to_pods:
            self.sku_to_pods[sku] = []
        self.sku_to_pods[sku].append(pod)

    def _available_similarity_pod(self, skus_in_station):

        return

    # Default
    # def get_available_pod(self, sku: str):
    #     if sku in self.sku_to_pods:
    #         for pod in self.sku_to_pods[sku]:
    #             if pod.is_idle is True:
    #                 return pod
                
    def get_available_pod_similarity(self, sku: str, skus_in_order, station_coordinate):
        # If SKU is available
        sku_in_order_list = [i for i in skus_in_order]
        pod_available_for_multiple_items = pd.DataFrame(columns=["pod_id", "similarity_score", "distance_to_station"])
        
        print("SKU IN ORDER")
        print(skus_in_order)

        station_coordinate = [station_coordinate.x, station_coordinate.y]

        if sku in self.sku_to_pods:
            for pod in self.sku_to_pods[sku]:
                similarity_score = 0

                if pod.is_idle is True:
                    pod_skus = [i for i in pod.skus]
                    pod_skus_in_station_skus_mask = np.isin(sku_in_order_list, pod_skus)
                    pod_skus_in_station_skus = np.array(sku_in_order_list)[pod_skus_in_station_skus_mask]
                    
                    if len(pod_skus_in_station_skus) > 0:
                        for skus in pod_skus_in_station_skus:
                            skus_qty_in_pod = pod.get_quantity(skus)
                            if skus_qty_in_pod > skus_in_order[skus]:
                                similarity_score += 1
                    
                    pod_coordinate = [pod.coordinate.x, pod.coordinate.y]
                    distance = manhattan_distances([pod_coordinate],[station_coordinate])[0][0]
                    pod_available_for_multiple_items = pd.concat([pod_available_for_multiple_items, 
                                                                pd.DataFrame([[pod.pod_id, similarity_score, distance]], 
                                                                                                            columns=["pod_id", 
                                                                                                                    "similarity_score", 
                                                                                                                    "distance_to_station"])], ignore_index=True) 
            pod_available_for_multiple_items["distance_score"] = pod_available_for_multiple_items["distance_to_station"].max() - pod_available_for_multiple_items["distance_to_station"]
            pod_available_for_multiple_items.sort_values(by=["similarity_score", "distance_score"], ascending=[False, False], inplace=True)
            pod_available_for_multiple_items.reset_index(drop=True, inplace=True)
            pod_available_for_multiple_items = pod_available_for_multiple_items[pod_available_for_multiple_items["similarity_score"] > 0]

            assigned_pod = None
            if len(pod_available_for_multiple_items) > 0:
                assigned_pod_id = pod_available_for_multiple_items.loc[0, "pod_id"]
           
                assigned_pod = self.get_pod_by_id(assigned_pod_id)
            
            print("ASSIGNED POD")
            print(assigned_pod)
            return assigned_pod

    def mark_pod_not_available(self, coordinate: NetLogoCoordinate):
        pod = self.coordinate_to_pods.get((coordinate.x, coordinate.y))
        pod.is_idle = False

    def mark_pod_available(self, coordinate: NetLogoCoordinate):
        pod = self.coordinate_to_pods.get((coordinate.x, coordinate.y))
        station = pod.station
        station.remove_pod(pod.pod_id)
        pod.is_idle = True

    def get_pods_by_sku(self, sku):
        return self.sku_to_pods.get(sku, None)

    def get_pod_by_coordinate(self, x, y):
        return self.coordinate_to_pods.get((x, y), None)

    def get_pod_by_id(self, pod_id):
        return self.id_to_pod.get(pod_id, None)
