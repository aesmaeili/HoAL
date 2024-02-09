"""Specifies an HL System Specifications"""

from holonic_ml_avg import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
import os
import torch.distributed as dist
from hl_configs import Globals

class HLAvg_System:
    def __init__(
        self,
        model,
        test_dataset,
        terminalhs,
        generator,
        sum_writer,
        nonterms_log_dir="", 
        terms_log_dir="",
        name="HL",
        synced_training=False,
        device=torch.device("cpu"),
        # max_message_tensor_size=Globals.max_message_tensor_size,
    ) -> None:
        self.terminalhs = terminalhs  # TODO:lets replace them with functions that extract terminals and nons from DF
        if isinstance(terminalhs, list):
            self.devices = terminalhs
        else:
            raise Exception("terminalhs: must be a list")
        self.DF = {}
        for th in self.terminalhs:
            self.DF[th.id] = th
        self.root = None
        self.test_dataset = test_dataset
        self.global_model = model
        self.generator = generator
        self.sum_writer = sum_writer
        self._trained = False
        
        self.processes = []
        self.nonterms_log_dir=nonterms_log_dir
        self.terms_log_dir=terms_log_dir
        self.name = name
        self.childQ = torch.multiprocessing.Queue() # a quueue to pass messages to the children processes
        self.max_message_tensor_size=Globals.max_message_tensor_size
        self.synced_training = synced_training
        self.device = device

    def reset_models(self):
        """resets the whatever the holons of the system have learned"""
        for h in self.DF.values():
            h.reset_model()
            if isinstance(h, TerminalMLAvgHolon):
                h.criterion = h.criterion.__class__()
                h.optimizer = h.optimizer.__class__(h.optimizer.param_groups, lr=h.optimizer.defaults["lr"])


    def reset_holarchy(self):
        """resets the vertical connections of the holons (clears superholons if neccessary)"""
        # print(f"resetting holarchy: DF contents: {self.DF}")
        for h in self.DF.copy().values():
            self.DF[h.id].superhs = {}
            self.DF[h.id].is_connected_to_superh = True  # by default all holons are connected to their superholons
            self.DF[h.id].subhs = {}
            self.DF[h.id].connected_subhs = {}
            self.DF[h.id].communications_count["vertical"] = 0
            if isinstance(self.DF[h.id], NonTerminalMLAvgHolon):
                self.DF.pop(h.id)
                # del self.DF[h.id]

    def reset_neighbors(self):
        """resets the neighborhood graphs of the holons"""
        for h in self.DF.values():
            h.reset_neighbors()
            h.communications_count["horizontal"] = 0

    def reset_everything(self):
        """self-explanatory!"""
        self.reset_models()
        self.reset_holarchy()
        self.reset_neighbors()
        self.force_terminate() # terminating the multiprocesses
    
    def set_log_dirs(self, terms_log_dir, nonterms_log_dir):
        self.nonterms_log_dir=nonterms_log_dir 
        self.terms_log_dir=terms_log_dir
        for h in self.DF.values():
            h.set_log_dir(terms_log_dir) if isinstance(h, TerminalMLAvgHolon) else h.set_log_dir(nonterms_log_dir)

    def set_sys_name(self, new_name):
        self.name = new_name
        for h in self.DF.values():
            h.set_sys_name(new_name)

    def _build_holarchy(self, structure, rank_start):
        """builds the holarchy based on the specified structure in form of
        nested lists of holon ids. The first item of each list is assumed to
        be a at least 1-item tuple containing the id of the list. This id is used as
        the id of the superholon. The second item if provided is the rank if it is of integer type
        and is the indication whether the superholon should be an aggregator if it is
        of boolean type. if the rank is not provided, it will be
        automatically set and if aggregation indication not provided, it will be assumed True. 
        This is the private function working in the backscene.
        """
        
        if not isinstance(structure, list):
            raise Exception("The structure must be a nested list")
        elif not isinstance(structure[0], tuple):
            raise Exception(f"Missing root name in list {structure}")
        else:
            if len(structure[0])==3:
                root = NonTerminalMLAvgHolon(structure[0][0],structure[0][1], {}, {}, {}, self.global_model, global_test_data=self.test_dataset, log_dir=self.nonterms_log_dir, sys_name=self.name, is_aggregator=structure[0][2], synced=self.synced_training, connected_subhs={}, device=self.device)
            elif len(structure[0])==2:
                if isinstance(structure[0][1], bool):
                    root = NonTerminalMLAvgHolon(structure[0][0],len(self.DF), {}, {}, {}, self.global_model, global_test_data=self.test_dataset, log_dir=self.nonterms_log_dir, sys_name=self.name, is_aggregator=structure[0][1], synced=self.synced_training, connected_subhs={}, device=self.device)
                elif isinstance(structure[0][1], int):
                    root = NonTerminalMLAvgHolon(structure[0][0],structure[0][1], {}, {}, {}, self.global_model, global_test_data=self.test_dataset, log_dir=self.nonterms_log_dir, sys_name=self.name, synced=self.synced_training, connected_subhs={}, device=self.device)
                else:
                    raise Exception(f"Unrecognized superholon tupple configuration {structure[0]}")
            elif len(structure[0])==1:
                root = NonTerminalMLAvgHolon(structure[0][0],len(self.DF), {}, {}, {}, self.global_model, global_test_data=self.test_dataset, log_dir=self.nonterms_log_dir, sys_name=self.name, synced=self.synced_training, connected_subhs={}, device=self.device)
            else:
                raise Exception(f"Unrecognized superholon tupple configuration {structure[0]}")
            self.DF[root.id] = root
            for i in structure[1:]:
                rank_start += 1
                if isinstance(i, str):
                    
                    root.connect_to_subh(self.DF[i], self.DF[i].weight)
                    self.DF[i].connect_to_super(root, root.weight)
                else:
                    subholarchy = self._build_holarchy(i, len(self.DF))
                    
                    root.connect_to_subh(subholarchy, subholarchy.weight)
                    subholarchy.connect_to_super(root, root.weight)
            return root

    def build_holarchy(self, structure):
        """The main function to be called"""
        ranks_start = len(self.terminalhs)
        self.root = self._build_holarchy(structure, ranks_start)
        for h in self.DF.values(): #updating the weights of the superholons
            for s in h.superhs.values():
                h.set_weight(s[0], self.DF[s[0].id].weight)

    def build_neighbors(self, graph):
        """builds the inter-holon interaction network based on the provided
        graph (list of 2- or 3-item tuples including the holon ids and the weights)
        i.e. (source,neighbor,weight) if 3 is provided. If weight is not provided it will
        use the default weight (here the dataset size). When 3 items are provided, it assumes
        a directional connection, and uses the weight on sources only. For bidirection communications
        one needs to manually specify the weight for the reciprocal interaction."""
        if not isinstance(graph, list):
            raise Exception("The graph must be a list of tuples")
        for t in graph:
            if not isinstance(t, tuple):
                raise Exception("The graph must be a list of tuples")
            elif len(t) < 2 or len(t) > 3:
                raise Exception(f"The tuples must be of size 3, yours is {len(t)}")
            elif len(t) == 3:
                self.DF[t[0]].add_neighbor(self.DF[t[1]], t[2])
                self.DF[t[1]].add_neighbor(self.DF[t[0]], 0)  
            else:
                self.DF[t[0]].add_neighbor(self.DF[t[1]], self.DF[t[1]].weight)
                self.DF[t[1]].add_neighbor(self.DF[t[0]], self.DF[t[0]].weight) # because of non-directional
    
    def manage_disconnecteds(self, dis_list):
        """ disconnects holons from their superholons based on the provided list of holon ids"""
        for h in self.DF.values():
            if h.id in dis_list or (h.superhs == {} and isinstance(h, NonTerminalMLAvgHolon)):
                for s in h.superhs.values():
                    h.disconnect_from_super(s[0].id)
                    s[0].disconnect_from_subh(h.id)

    def export_graph(self):
        """exports the holarchy as a networkx graph"""
        G = nx.Graph()
        for hn, h in self.DF.items():
            if isinstance(h, NonTerminalMLAvgHolon):
                if h.is_connected_to_superh:
                    G.add_node(hn, ntype="non-term-connected")
                else:
                    G.add_node(hn, ntype="non-term-disconnected")
            else:
                if h.is_connected_to_superh:
                    G.add_node(hn, ntype="term-connected")
                else:
                    G.add_node(hn, ntype="term-disconnected")
        for h in self.DF.values():
            G.add_edges_from(
                h.get_neighbor_connections_list(),
                weight=0.75,
                color="tab:gray",
                style="dashed",
            )
            if isinstance(h, NonTerminalMLAvgHolon):
                G.add_edges_from(
                    h.get_all_subs_list(), weight=1, color="tab:gray"
                )

        return G

    def visualize_holarchy(self, with_labels, in_file=None):
        G = self.export_graph()
        node_colors = []
        node_sizes = []
        for u in G.nodes(data="ntype"):
            if u[1] == "non-term-connected":
                node_colors.append("tab:red")
                node_sizes.append(500)
            elif u[1] == "non-term-disconnected":
                node_colors.append("tab:purple")
                node_sizes.append(500)
            elif u[1] == "term-connected":
                node_colors.append("tab:blue")
                node_sizes.append(300)
            elif u[1] == "term-disconnected":
                node_colors.append("tab:green")
                node_sizes.append(300)
            else:
                raise Exception("Unrecognized node type")
        
        pos = nx.drawing.nx_agraph.graphviz_layout(
            G, prog="twopi", root=self.root.id, args=""
        )
        plt.figure(figsize=(8, 8))
        nx.draw_networkx(
            G,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8,
            with_labels=with_labels,
            edgecolors="gray",
        )
        plt.tight_layout()
        plt.axis("off")
        if not in_file is None:
            plt.savefig(in_file, format="pdf", bbox_inches="tight")
        else:
            plt.show()

    def _start_holon(self, holon, msgQ, rounds, runs):
        """initializes the holons in distributed environment"""
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["WORLD_SIZE"] = str(len(self.DF))
       
        dist.init_process_group(backend="gloo", world_size=len(self.DF), rank=holon.rank)
        holon.train(msgQ, rounds, runs, self.max_message_tensor_size)

    def train_system(self, rounds, runs):
        """initializes the training process of the holons distributedly"""
        print(f"Starting the distributed training process of {self.name} system...")
        processes = []
        for h in self.DF.values():
            
            p = torch.multiprocessing.Process(target=self._start_holon, args=(h,self.childQ, rounds, runs))
            p.start()
            processes.append(p)
        self.processes = processes
        i = 0
        num_process = len(processes)
        while i < num_process:
            if not self.childQ.empty():
                self.childQ.get()
                print(f"System- a process just finished- {i+1}/{num_process}")
                i += 1
        self.force_terminate()
       
        print("Finished training the holons distributedly")
    
    def force_terminate(self):
        for p in self.processes:
            p.terminate()

    