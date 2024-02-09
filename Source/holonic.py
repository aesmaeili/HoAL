"""Include the definitions related to holonic systems and the ML models they are working on """
from hl_configs import Globals

class Holon(object):
    def __init__(
        self,
        id,
        neighbors,
        subhs,
        connected_subhs,
        superhs,
        is_connected_to_superh,
    ) -> None:
        self.id = id
        self.neighbors = neighbors
        self.subhs = subhs
        self.connected_subhs = connected_subhs
        self.superhs = superhs
        self.is_connected_to_superh = is_connected_to_superh


    def _is_root(self) -> bool:
        return len(self.superhs) == 0 and len(self.subhs) > 0

    def add_neighbor(self, n, w):
        if not n.id in list(self.neighbors.keys()):
            self.neighbors[n.id]=(n, w)
    
    def remove_neighbor(self, n):
        if n.id in list(self.neighbors.keys()):
            del self.neighbors[n.id]

    def reset_neighbors(self):
        self.neighbors = dict()

    
    def add_super(self, sup, w):
        if isinstance(sup, NonTerminalHolon):
            self.superhs[sup.id]=(sup, w)
        else:
            raise Exception(f"The super holon must be nonterminal, but you provided {type(sup).__name__}")
        
    def connect_to_super(self, sup, w):
        if sup.id not in list(self.superhs.keys()):
            self.add_super(sup, w)
        self.is_connected_to_superh=True
    
    def disconnect_from_super(self, sup_id): # do not remove the super from the list of supers
        if sup_id in list(self.superhs.keys()):
            self.is_connected_to_superh=False
    
    def set_weight(self, peer, w):
        if peer.id in (self.superhs.keys()):
            self.superhs[peer.id]=(self.superhs[peer.id][0], w)
        elif peer.id in (self.subhs.keys()):
            self.subhs[peer.id]=(self.subhs[peer.id][0], w)
        elif peer.id in (self.neighbors.keys()):
            self.neighbors[peer.id]=(self.neighbors[peer.id][0], w)

    def leave_super(self, sup_id):
        if sup_id in (self.superhs.keys()):
            self.superhs.pop(sup_id)
            self.is_connected_to_superh=False


    def get_neighbor_connections_list(self):
        l = []
        for n in self.neighbors.keys():
            l.append((self.id, n))
        return l
    

    def __str__(self) -> str:
        return self.id


class NonTerminalHolon(Holon):
    def __init__(
        self,
        id,
        neighbors,
        subhs,
        connected_subhs,
        superhs,
        is_connected_to_superh,
    ) -> None:
        super().__init__(id=id, neighbors=neighbors, subhs=subhs, connected_subhs=connected_subhs, superhs=superhs, is_connected_to_superh=is_connected_to_superh)
        Globals.current_num_nonterminal_holons += 1
    
    def add_subh(self, sub, w):
        self.subhs[sub.id]=(sub, w)
        # sub.add_super(self, w)
    
    def connect_to_subh(self, sub, w):
        if sub.id not in list(self.subhs.keys()):
            self.add_subh(sub, w)
        self.connected_subhs[sub.id]=(sub, w)

    def disconnect_from_subh(self, sub_id): # does not remove the sub from the list of subs completely
        if sub_id in list(self.connected_subhs.keys()):
            self.connected_subhs.pop(sub_id)
    
    def remove_subh(self, sub_id):
        if sub_id in list(self.subhs.keys()):
            self.subhs.pop(sub_id)
            self.connected_subhs.pop(sub_id)
    
    def get_all_subs_list(self):
        l = []
        for s in self.subhs.keys():
            l.append((self.id, s))
        return l
    
    def get_connected_subs_list(self):
        l = []
        for s in self.connected_subhs.keys():
            l.append((self.id, s))
        return l


class TerminalHolon(Holon):
    def __init__(
        self,
        id,
        neighbors,
        superhs,
        is_connected_to_superh,
    ) -> None:
        super().__init__(id=id, neighbors=neighbors, subhs=[], connected_subhs={}, superhs=superhs, is_connected_to_superh=is_connected_to_superh)
        Globals.current_num_terminal_holons += 1

    
