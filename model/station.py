from typing import List

from engine.object import Object
from engine.netlogo_coordinate import NetLogoCoordinate
from .order_manager import OrderManager


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
        self.max_orders = 2
        self.skus = {}
        super().__init__()

    def add_order(self, order_id: int):
        self.order_ids.append(order_id)

    def remove_order(self, order_id: int):
        self.order_ids.remove(order_id)

    def is_picker_station(self) -> bool:
        return self.station_type == "picker"

    def is_replenishment_station(self) -> bool:
        return self.station_type == "replenishment"

    def get_skus_in_station(self, order_manager: OrderManager):
        self._skus_in_station(order_manager)
        return self.skus
    
    def _skus_in_station(self, order_manager: OrderManager):
        for order_id in self.order_ids:
            order = order_manager.get_order_by_id(order_id)
            print("Remaining SKU")
            print(order.get_remaining_skus())
            for sku, value in order.get_remaining_skus().items():
                if sku not in self.skus:
                    self.skus[sku] = value
                self.skus[sku] += value
        return
