""" Testing the HLAvg system"""
# import torch.multiprocessing as mp
import torch
import torch.nn as nn
from models import *
import numpy as np
import time
import datetime
import utilities
import shutil


from hlavg_system import *
from hl_configs import Globals




""" Logging configs"""
# utilities.config_logger(logger, "hl_logs.log", "INFO")


if __name__ == "__main__":
    shutil.rmtree("Experiments")
    os.makedirs("Experiments")

 
    """Configing the multiprocessing environment"""
    torch.multiprocessing.set_start_method("spawn")

    """Configing the tensorboard writer object"""

    """ Attempting to make the results reproducable"""
    g = torch.Generator()
    g.manual_seed(0)
    np.random.seed(0)

    experiment_id = "1"

    """ configing the device to be used"""
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("cuda is available")
        device = torch.device("cuda:0")
        # torch.cuda.set_device(0)
    elif torch.backends.mps.is_available():
        print("cuda is not available but mps is")
        device = torch.device("mps")
    
    print(f"device: {device}")

    """ Some initializations """
    Globals.init()
    synced_training = True

    """the same model to start for all holons"""
    start_model = CNNMnist()
    #start_model = MLP()
    # start_model = MNISTModel()
    Globals.max_message_tensor_size = (
        Message("dummy", start_model.state_dict).get_tensor_size()[0] + 1000
    )

    """creating the holons"""
    each_holon_training_params = {
        "num_epochs": 5,
        "batch_size": 32,
        "criterion": torch.nn.CrossEntropyLoss,
        "optimizer": torch.optim.SGD,
        "optimizer_args": {"lr": 0.01},
        "synced": synced_training,
    }
    data_params = {
        "dataset_name": "mnist",
        "samp_type": "iid",
        "unequal_splits": False,
    }


    all_systems_structures = {
        "HL2L": {
            "holarchy": [
                ("R",),
                [("H1",), "0", "1", "2", "3", "4"],
                [("H2",), "5", "6", "7", "8", "9"],
            ],
            "neighborhood": [("0", "1"),("0", "3"),("1", "4"),("2", "3"),
                             ("5", "6"),("5", "8"),("6", "9"),("7", "8"),
                             ("H1", "H2"),
            ],
            "disconnecteds": [],
        },
        "HL3L": {
            "holarchy": [
                ("R",),
                [("H1",), 
                    [("H3",),"0", "1", "2"],
                    [("H4",), "3", "4"]
                ],
                [("H2",), 
                    [("H5",),"5", "6", "7"],
                    [("H6",), "8", "9"]
                ],
            ],
            "neighborhood": [("0", "1"),("0", "3"),("1", "4"),("2", "3"),
                             ("5", "6"),("5", "8"),("6", "9"),("7", "8"),
                             ("H1", "H2"),
            ],
            "disconnecteds": [],
        },
        "HL4L": {
            "holarchy": [("R",),
                [("H1",), 
                    [("H3",),[("H5",),"0", "1"],[("H6",),"2", "3"],],
                    [("H4",),[("H7",),"4", "5"],[("H8",),"6", "7"],]
                ],
                [("H2",),"8", "9"]
            ],
            "neighborhood": [("0", "1"),("0", "3"),("1", "4"),("2", "3"),
                             ("5", "6"),("5", "8"),("6", "9"),("7", "8"),
                             ("H1", "H2"),
            ],
            "disconnecteds": [],
        },
    }


    terminals, global_test_data = utilities.create_homogen_terminal_holons(
        10, MLP, starting_model=start_model, **each_holon_training_params, **data_params
    )

    #
    num_of_trials = 1
    plots_dir = os.path.join(os.path.curdir, "Plots")

    
    systems_to_try = ["HL2L", "HL3L", "HL4L"]
    for t in range(num_of_trials):
        for s_name, s_struct in utilities.extract_dic_from_keys(
            all_systems_structures, systems_to_try
        ).items():
            start_time = time.time()
            
            HS = HLAvg_System(CNNMnist, global_test_data, terminals, None, None, synced_training=synced_training, device=device)
            

            """my own performance logs"""
            trial_log_dir = os.path.join(os.path.curdir, "Experiments", s_name)
            
            trial_terms_path = os.path.join(trial_log_dir, str(t + 1), "terminals")
            trial_nonterms_path = os.path.join(
                trial_log_dir, str(t + 1), "nonterminals"
            )
            os.makedirs(trial_terms_path)
            os.makedirs(trial_nonterms_path)
            HS.build_holarchy(s_struct["holarchy"])
            HS.build_neighbors(s_struct["neighborhood"])
            HS.manage_disconnecteds(s_struct["disconnecteds"])
            HS.set_sys_name(s_name)
            HS.set_log_dirs(trial_terms_path, trial_nonterms_path)
            HS.visualize_holarchy(
                with_labels=True,
                in_file=os.path.join(plots_dir, f"holarchy_{s_name}.pdf"),
            )
            try:
                HS.train_system(2, 50)
            except KeyboardInterrupt:
                HS.force_terminate()
                raise SystemExit("Training quitted by the user.")
            end_time = time.time()
            print(f"Time elapsed for {s_name}: {end_time - start_time}")
            HS.reset_everything()  # preparing for the next system structure and train
            # TODO: each time build_holarchy is called, new superholons are created. 
            time.sleep(1)

