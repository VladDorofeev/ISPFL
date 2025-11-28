from ..fedavg.fedavg import FedAvg
from .oort_server import OortServer
from .oort_client import OortClient


class Oort(FedAvg):
    def __init__(self, num_clients_subset):
        super().__init__(num_clients_subset)

    def _init_server(self, cfg):
        self.server = OortServer(cfg)

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = OortClient
        self.client_kwargs["client_cls"] = self.client_cls