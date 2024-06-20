from typing import List, Optional

from engine.object import Object
from engine.netlogo_coordinate import NetLogoCoordinate
from .order_manager import OrderManager
from .pod import Pod
from .order import Order


class Station(Object):
    def __init__(self, station_id: int, station_type: str):
        self.station_id = f"{station_type}-{station_id}"
        self.station_type = station_type
        self.shape = 'empty-space'
        self.object_type = 'station'
        self.mass = 1
        self.coordinate = None
        self.path: List[NetLogoCoordinate] = []
        self.order_ids: List[int] = []
        self.max_orders = 6
        self.skus = {}
        self.incoming_pod: List[int] = []
        super().__init__()

    def add_order(self, order_id: int):
        self.order_ids.append(order_id)

    def remove_order(self, order_id: int):
        if order_id in self.order_ids:
            self.order_ids.remove(order_id)

    def is_picker_station(self) -> bool:
        return self.station_type == "picker"

    def is_replenishment_station(self) -> bool:
        return self.station_type == "replenishment"
    
    def add_pod(self, pod):
        self.incoming_pod.append(pod)
    
    def remove_pod(self, pod):
        if pod in self.incoming_pod:
            self.incoming_pod.remove(pod)

    def get_skus_in_station(self, order_manager: OrderManager):
        self._skus_in_station(order_manager)
        return self.skus
    
    def _skus_in_station(self, order_manager: OrderManager):
        for order_id in self.order_ids:
            order = order_manager.get_order_by_id(order_id)
            for sku, value in order.get_remaining_skus().items():
                if sku not in self.skus:
                    self.skus[sku] = value
                self.skus[sku] += value
        return
    
    def get_orders_in_station(self, order_manager: OrderManager):
        orders = []
        for order_id in self.order_ids:
            order = order_manager.get_order_by_id(order_id)
            orders.append(order)
        return orders
    
    def get_orders_in_station(self, order_manager: OrderManager) -> Optional[List[Order]]: 
        orders = []
        for order_id in self.order_ids:
            order = order_manager.get_order_by_id(order_id)
            orders.append(order)
        return orders

