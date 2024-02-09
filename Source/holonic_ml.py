"""Defines the specifications of the Holonic ML system"""

from holonic import TerminalHolon, NonTerminalHolon
from torch.utils.data import DataLoader
from data_tools import DatasetSplit
import random
import copy


class TerminalMLHolon(TerminalHolon):
    def __init__(
        self,
        id,
        neighbors,
        superhs,
        ml_model,
        data,
        num_epochs,
        batch_size,
        starting_model=None,
        data_idxs={},
        global_test_data=None,
        generator=None,
        synced=False,
        is_connected_to_superh=True,
        **kwargs,
    ) -> None:
        super().__init__(id=id, neighbors=neighbors, superhs=superhs, is_connected_to_superh=is_connected_to_superh)
        self.ml_model = ml_model
        self.starting_model = starting_model
        if not starting_model is None:
            self.model=copy.deepcopy(self.starting_model)
        else:
            self.model = self.ml_model()
            self.starting_model = copy.deepcopy(self.model)
        self.data = data
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.criterion = None
        self.optimizer = None
        self.lr = None
        self.data_idxs=data_idxs
        self.generator = generator
        self.__dict__.update(kwargs)
        if len(self.data_idxs) > 0:
            (
                self.train_loader,
                self.valid_loader,
                self.test_loader,
            ) = self.train_val_test()
        else:
            self.train_loader = DataLoader(
                self.data, self.batch_size, shuffle=True, generator=self.generator
            )
        self.has_global_test_data = not global_test_data is None
        if self.has_global_test_data:
            self.global_test_data_loader = DataLoader(
                global_test_data, generator=self.generator
            )

        if hasattr(self.data, "classes"):
            self.class_labels=list(range(len(self.data.classes)))
        else:
            self.class_labels=None
        
        if not self.class_labels is None:
            self.class_counts = [0] * len(self.class_labels)
            for _, y in self.train_loader:
                for yy in y:
                    self.class_counts[yy.item()] += 1
        
        self.communications_count = {"horizontal":0, "vertical":0}
        self.synced = synced
    

    def reset_model(self):
        self.model.load_state_dict(self.starting_model.state_dict())
        # self.model.load_state_dict(self.ml_model().state_dict())
    
    def train_val_test(self, tr_spl=0.8, vl_spl=0.1, ts_spl=0.1):
        """
        Returns train, validation and test dataloaders for a given dataset
        and device indexes.
        """
        dataset = self.data
        idxs = list(self.data_idxs)
        random.shuffle(idxs) # to shuffle the order of items

        idxs_train = idxs[: int(tr_spl * len(idxs))]
        idxs_val = idxs[int(tr_spl * len(idxs)) : int((tr_spl+vl_spl) * len(idxs))]
        idxs_test = idxs[int((tr_spl+vl_spl) * len(idxs)) :]
        
        self.train_data_size = len(idxs_train)
        self.test_data_size = len(idxs_test)
        self.valid_data_size = len(idxs_val)

        trainloader = DataLoader(
            DatasetSplit(dataset, idxs_train),
            batch_size=self.batch_size,
            shuffle=True,
            generator=self.generator,
        )
        validloader = DataLoader(
            DatasetSplit(dataset, idxs_val),
            batch_size=int(len(idxs_val) / 10),
            shuffle=False,
            generator=self.generator,
        )
        testloader = DataLoader(
            DatasetSplit(dataset, idxs_test),
            batch_size=int(len(idxs_test) / 10),
            shuffle=False,
            generator=self.generator,
        )

        return trainloader, validloader, testloader


class NonTerminalMLHolon(NonTerminalHolon):
    def __init__(
        self,
        id,
        neighbors,
        subhs,
        connected_subhs,
        superhs,
        ml_model,
        data,
        global_test_data=None,
        starting_model=None,
        is_aggregator=True,
        synced=False,
        is_connected_to_superh=True,
        **kwargs,
    ) -> None:
        super().__init__(id=id, neighbors=neighbors, subhs=subhs, connected_subhs=connected_subhs, superhs=superhs, is_connected_to_superh=is_connected_to_superh)
        self.ml_model = ml_model
        self.starting_model = starting_model
        if self.starting_model is None:
            self.model = self.ml_model()
            self.starting_model = copy.deepcopy(self.model)
        else:
            self.model = copy.deepcopy(self.starting_model)
        self.data = data
        self.has_global_test_data = not global_test_data is None
        if self.has_global_test_data:
            self.global_test_data_loader = DataLoader(
                global_test_data
            )
        self.__dict__.update(kwargs)
        self.class_counts=[]
        self.is_aggregator = is_aggregator
        self.has_aggregated = False
        self.communications_count = {"horizontal":0, "vertical":0}
        self.synced = synced
    
    def reset_model(self):
        if self.starting_model is None:
            self.model = self.ml_model()
        else:
            self.model.load_state_dict(self.starting_model.state_dict())
