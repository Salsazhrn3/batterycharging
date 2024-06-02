import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class Zone:
    boundaries = [] # 3 Dimension Array berarti appendnya 2D
    penalty = [] # 1D Array of integer
    cluster_num = 2

    def __init__(self, robots_location,methods):
        if methods == "default":
            self.boundaries = [
        [
            [10,5],
            [0,19],
        ],
        [
            [10,20],
            [0,29],
        ],
        [
            [10,30],
            [0,43],
        ],
        [
            [20,5],
            [11,19],
        ],
        [
            [20,20],
            [11,29],
        ],
        [
            [20,30],
            [11,43],
        ],
        [
            [30,5],
            [21,19],
        ],
        [
            [30,20],
            [21,29],
        ],
        [
            [30,30],
            [21,43],
        ]
        ]
        elif methods == "kmeans":
            self.kmeans_clustering(robots_location)
        
    def get_boundary(self):
        return self.boundaries
    
    def calculate_penalty(self, robots_location):
        self.penalty = [1] * len(self.boundaries)
        for robot in robots_location:
            for index, zone in enumerate(self.boundaries):
                if ((robot[1] <= zone[0][0] and robot[1] >= zone[1][0]) and (robot[0] >= zone[0][1] and robot[0] <= zone[1][1])):
                    self.penalty[index] += 1
        return self.penalty
    
        
    def _silhouette_score(self, robots_location, min_cluster, max_cluster):
        inertias = []
        silhouette_scores = []
        for k in range(min_cluster, max_cluster + 1):
            kmeans = KMeans(n_clusters=k, random_state=0).fit(robots_location)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(robots_location, kmeans.labels_))
        
        best_cluster = range(min_cluster, max_cluster + 1)[silhouette_scores.index(max(silhouette_scores))]
        return best_cluster
    
    @staticmethod
    def _minimum_bounding_rectangle(points):
        x_coords, y_coords = zip(*points)
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        if x_max - x_min + 1 < 3:
            diff = 3 - (x_max - x_min + 1)
            x_max += diff // 2
            x_min -= diff - (diff // 2)
        if y_max - y_min + 1 < 6:
            diff = 6 - (y_max - y_min + 1)
            y_max += diff // 2
            y_min -= diff - (diff // 2)
        
        return [[y_max, x_min], [y_min, x_max]]
    
    def kmeans_clustering(self, robots_location):
        """Clustering using KMeans

        Args:
            robots_location (list): list of robots location.
            min_col (int): Minimum column for a zone.
            min_row (int): Minimum row for a zone.

        Returns:
            lists of zone boundaries
        """
        robots = np.array(robots_location)
        if len(robots) != 0:
            self.cluster_num = self._silhouette_score(robots, min_cluster=2, max_cluster=9)
            kmeans = KMeans(n_clusters=self.cluster_num, random_state=0)
            labels = kmeans.fit_predict(robots)
            boundaries = []
            for cluster_id in range(self.cluster_num):
                cluster_points = robots[labels == cluster_id]
                print("Cluster points")
                print(cluster_points)
                cluster_boundary_points = self._minimum_bounding_rectangle(cluster_points)
                boundaries.append(cluster_boundary_points)

            self.boundaries = boundaries
        return
    def affinity_propagation():
        return 
    
    def route_clustering():
        return

