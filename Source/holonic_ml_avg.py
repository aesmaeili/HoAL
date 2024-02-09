"""Defines the specifications of the Holonic AVG ML system"""

from queue import Queue
from holonic_ml import TerminalMLHolon, NonTerminalMLHolon

import torch
import torch.nn as nn
import torch.distributed as dist

from messages import Message
import copy
import datetime
import time
import threading
from threading import Event
from itertools import chain
import utilities
import os
import math
from hl_configs import Globals


class TerminalMLAvgHolon(TerminalMLHolon):
    def __init__(
        self,
        id,
        rank,  # for torch.distributed procees use
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
        device=None,
        log_dir="",
        sys_name="HL",
        synced=False,
        is_connected_to_superh=True,
        **kwargs,
    ) -> None:
        super().__init__(
            id=id,
            neighbors=neighbors,
            superhs=superhs,
            ml_model=ml_model,
            data=data,
            num_epochs=num_epochs,
            batch_size=batch_size,
            starting_model=starting_model,
            data_idxs=data_idxs,
            global_test_data=global_test_data,
            generator=generator,
            synced=synced,
            kwargs=kwargs,
            is_connected_to_superh=is_connected_to_superh,
        )
        self.rank = rank
        self.device = device
        self.received_neighbor_messages = dict()
        self.received_sup_messages = dict()
        self.received_req_messages = dict()
        self.weight = len(data_idxs)  # holon's self-importance weight
        # utilities.log_print(f"holon({self.id}): my weight is {self.weight}")
        self.message_size = Message("dummy", self.model.state_dict()).get_tensor_size()
        self.should_quit = False
        self.quit_reason = "Unspecified"
        self.global_run = 0
        self.log_dir = os.path.join(log_dir, self.id)
        self.sys_name = sys_name
        self.run_logs = {
            "train": {},
            "test": {},
            "system": self.sys_name,
            "data_dist": self.class_counts,
            "communication": [{"horizontal": 0, "vertical": 0}],
        }
        self.should_pause = False

    def send_msg(self, msg_tensor, dst, dst_id, timeout=datetime.timedelta(seconds=30)):
        msg_size_tensor = torch.tensor([msg_tensor.size()[0]])
        if msg_size_tensor.item() >= Globals.max_message_tensor_size:
            utilities.log_print(
                f"holon({self.id}): msg size is {msg_size_tensor.item()} and max is {Globals.max_message_tensor_size}"
            )
            raise Exception(
                f"holon({self.id}):the message will not fit in the tensor to be sent. Increase max_message_tensor_size"
            )
        else:
            zero_pad_tensor = torch.zeros(
                Globals.max_message_tensor_size - msg_size_tensor.item() - 1
            )
            augmented_msg_tensor = torch.concat(
                [msg_size_tensor, msg_tensor, zero_pad_tensor]
            )
            if dst_id in list(chain(self.neighbors.keys(), self.superhs.keys())):
                utilities.log_print(
                    f"holon({self.id}): sending a message to holon({dst_id})"
                )
                req = dist.isend(augmented_msg_tensor, dst)
                try:
                    req.wait(timeout)
                    utilities.log_print(
                        f"holon({self.id}): sent a message to holon({dst_id})"
                    )
                except:
                    utilities.log_print(
                        f"holon({self.id}): couldn't send a message to holon({dst_id})"
                    )
                    pass

    def recv_msg(self, src=None, timeout=datetime.timedelta(seconds=1800)):
        whole_msg = torch.zeros(Globals.max_message_tensor_size)
        # dist.recv(whole_msg, src)
        req = dist.irecv(whole_msg, src)
        req.wait(timeout)
        actual_msg_size = int(whole_msg[0].item())
        actual_msg_tensor = whole_msg[1 : actual_msg_size + 1].to(
            torch.ByteTensor().dtype
        )  # because the message initially serialiez into byte tensor
        return actual_msg_tensor

    def listen_to_messages(self, pause_event, quit_event):
        # utilities.log_print(f"holon({self.id}): messag size={self.message_size}")
        """An idea for the future. For now I am using pytorchs distributed library"""
        # event.clear()
        rcevd_msg_queue = Queue()
        process_msg_event = Event()
        process_msg_thread = threading.Thread(
            target=self.process_messages,
            args=(rcevd_msg_queue, pause_event, quit_event, process_msg_event),
        )
        process_msg_thread.start()
        while not self.should_quit:
            try:
                rcvd_msg = self.recv_msg()
                rcevd_msg_queue.put(rcvd_msg)
                process_msg_event.set()
            except RuntimeError:
                self.should_quit = True
                self.quit_reason = "Not received any meassages for a while"
                rcevd_msg_queue.put(None)
                process_msg_event.set()
                process_msg_thread.join()
                quit_event.set()
        rcevd_msg_queue.put(None)
        process_msg_event.set()
        process_msg_thread.join()
        quit_event.set()

    def log_stats(self, interval):
        """Logs the stats of the holon in the given interval
        for now it only logs the number of communications"""
        while not self.should_quit:
            self.run_logs["communication"].append(self.communications_count)
            time.sleep(interval)

    def set_train_params(self, criterion, optimizer):
        self.criterion = criterion
        self.optimizer = optimizer

    def set_log_dir(self, log_dir):
        self.log_dir = os.path.join(log_dir, self.id)
        # os.makedirs(self.log_dir)

    def set_sys_name(self, new_name):
        self.sys_name = new_name
        self.run_logs = {
            "train": {},
            "test": {},
            "system": self.sys_name,
            "data_dist": self.class_counts,
            "communication": [{"horizontal": 0, "vertical": 0}],
        }

    def process_messages(
        self, rcevd_msg_queue, pause_event, quit_event, process_msg_event
    ):
        while True:
            # process_msg_event.wait()
            process_msg_event.clear()
            recved_msg = rcevd_msg_queue.get()
            if recved_msg is None:
                break
            self._process_messages(recved_msg, pause_event, quit_event)

    def _process_messages(self, message_tensor, pause_event, quit_event):
        message = Message.from_tensor(message_tensor)
        sender_id = message["msg_from"]
        utilities.log_print(
            f"holon({self.id}): received a ({message['msg_type']}) message from holon({sender_id}). my pause flag is {self.should_pause} and last run is {self.global_run-1}"
        )
        if (
            message["msg_type"] == "hint"
            and set(self.superhs.keys())
            and set(self.neighbors[sender_id][0].superhs.keys())
        ):
            self.received_neighbor_messages[sender_id] = (
                message["msg_content"],
                self.neighbors[sender_id][1],
            )
            self.communications_count["horizontal"] += 1
            if not self.is_connected_to_superh and not pause_event.is_set():
                utilities.log_print(f"holon({self.id}): waking up by a hint")
                self.should_pause = False
                pause_event.set()
        elif message["msg_type"] == "update":
            self.received_sup_messages[sender_id] = (
                message["msg_content"],
                self.superhs[sender_id][1],
            )
            # self.should_pause = False
            utilities.log_print(f"holon({self.id}): setting the pause flag to true")
            pause_event.set()
            self.communications_count["vertical"] += 1
        elif (
            message["msg_type"] == "request"
            and set(self.superhs.keys())
            and set(self.neighbors[sender_id][0].superhs.keys())
        ):
            who_tuple = (
                self.neighbors[sender_id]
                if sender_id in list(self.neighbors.keys())
                else self.superhs[sender_id]
            )
            self.received_req_messages[sender_id] = (
                message["msg_content"],
                who_tuple[1],
            )
        elif message["msg_type"] == "continue":
            utilities.log_print(f"holon({self.id}): setting the pause flag to true")
            pause_event.set()
        elif message["msg_type"] == "quit":
            if set(self.superhs.keys()) and set(
                self.neighbors[sender_id][0].superhs.keys()
            ):
                self.received_neighbor_messages[sender_id] = (
                    message["msg_content"],
                    self.neighbors[sender_id][1],
                )
            self.remove_neighbor(self.neighbors[sender_id][0])
            self.communications_count["horizontal"] += 1
        else:
            pass  # discards the message

    def received_enough_hints(self):
        insiders_num = 0
        for k, v in self.neighbors.copy().items():
            if set(self.superhs.keys()) and set(
                self.neighbors[k][0].superhs.keys()
            ):  # the neighbor belongs to the same superholon
                insiders_num += 1
                if k not in list(self.received_neighbor_messages.keys()) and k in list(
                    self.neighbors.keys()
                ):
                    return False
        return len(self.received_neighbor_messages) > 0 or insiders_num == 0

    def received_enough_updates(self):
        for k, v in self.superhs.copy().items():
            if not k in list(self.received_sup_messages.keys()) or k not in list(
                self.superhs.keys()
            ):
                return False
        return len(self.received_sup_messages) > 0

    def aggregate_models(self, which_models):
        s_dicts = []
        n_ws = []
        w_offset = 0
        if which_models == "neighbors":
            while len(self.received_neighbor_messages) > 0:
                k, v = self.received_neighbor_messages.popitem()
                s_dicts.append(v[0])
                n_ws.append(v[1])
                w_offset = (
                    self.weight
                )  # TODO: if the holon is not connected to a superholon, then the weight should be 0 because its model is one round behind the neighbors

        elif which_models == "supers":
            while len(self.received_sup_messages) > 0:
                k, v = self.received_sup_messages.popitem()
                s_dicts.append(v[0])
                n_ws.append(v[1])
                utilities.log_print(
                    f"holon({self.id}): has a message from superholon({k}) with weight {v[1]}. superholon weight I have in database is {self.superhs[k][1]}"
                )
                # w_offset = self.weight  # experimenting

            if len(self.received_sup_messages) > 0:
                while len(self.received_sup_messages) > 0:
                    k, v = self.received_sup_messages.popitem()
                    s_dicts.append(v[0])
                    n_ws.append(v[1])
            else:
                while len(self.received_neighbor_messages) > 0:
                    k, v = self.received_neighbor_messages.popitem()
                    s_dicts.append(v[0])
                    n_ws.append(v[1])
                    w_offset = self.weight

        else:
            raise Exception(
                f"Unknown (unimplemented) aggregation models {which_models}"
            )

        if len(s_dicts) == 0:
            utilities.log_print(f"holon({self.id}): no received models to aggregate")
            return

        agg_state_dict = copy.deepcopy(self.model.state_dict())
        for pk, pv in agg_state_dict.items():
            if w_offset == 0:
                agg_state_dict[pk] = torch.zeros(pv.size())  # initialized with 0
            else:
                agg_state_dict[pk] = torch.mul(
                    w_offset, pv
                )  # initialized with my tensor. can support the above statement too, but added for clarity
            for i, s in enumerate(s_dicts):

                agg_state_dict[pk] += torch.mul(n_ws[i], s[pk]).to(self.device)
            agg_state_dict[pk] /= w_offset + sum(n_ws)
        self.model.load_state_dict(agg_state_dict)
        utilities.log_print(
            f"holon({self.id}):finished the aggregation of {which_models}. The weights were {n_ws}, and the offset was {w_offset}"
        )

    def save_logs(self, run="0"):
        utilities.save_dict_to_pickle(self.run_logs, self.log_dir, f"run_{run}")

    def terminate(self):
        self.save_logs(str(self.global_run))

    def train(self, msgQ, rounds, runs, max_message_tensor_size):
        Globals.max_message_tensor_size = max_message_tensor_size
        print(f"holon({self.id}): starting the training process....")
        utilities.log_print(
            f"holon({self.id}): is_connected_to_superh={self.is_connected_to_superh}"
        )

        os.makedirs(self.log_dir)
        pause_event = Event()
        quit_event = Event()
        rcv_thread = threading.Thread(
            target=self.listen_to_messages, args=(pause_event, quit_event)
        )
        log_thread = threading.Thread(target=self.log_stats, args=(10,))
        rcv_thread.start()
        log_thread.start()
        training_loss = 0
        training_correct = 0
        training_samples = 0
        global_steps = 0
        total_rounds = 1  # rounds # for now we are not using rounds for terminals
        self.global_run = 1
        while not self.should_quit and self.global_run <= runs:

            # utilities.log_print(f"holon({self.id}): waiting for update from a super")
            if not self.should_pause:
                current_round = 1
                round_stats = {}
                round_test_stats = {}
                # round_stats = {}
                run_stats = {self.global_run: {}}
                while current_round <= total_rounds:
                    epoch_stats = {}
                    epoch_test_stats = {}
                    round_stats[current_round] = {}
                    round_test_stats[current_round] = {}

                    if self.is_connected_to_superh and self.received_enough_updates():
                        utilities.log_print(
                            f"holon({self.id}): received enough updates, so aggregating them"
                        )
                        self.aggregate_models("supers")
                        self.received_neighbor_messages = (
                            dict()
                        )  # to discard the hints received in the previous round

                    elif self.received_enough_hints():  # and not (

                        """the last round of the last run is not included in the aggregation because
                        the holon will quit after that round and the model will be sent to the super holon
                        without being tuned. Updated: the last round of the last run is included in the aggregation because
                        I have moved the aggregation process to the beginning of training process
                        """
                        utilities.log_print(
                            f"holon({self.id}): received enough hints, so aggregating them"
                        )
                        self.aggregate_models("neighbors")
                        """reseting the received messages for the next collaboration round"""

                    """training the model"""
                    """moving the model to the device"""
                    self.model.to(self.device)
                    for e in range(self.num_epochs):
                        utilities.log_print(f"holon({self.id}): starting epoch {e+1}")
                        training_stats = {}
                        for i, (X, y) in enumerate(self.train_loader):
                            X = X.to(self.device)
                            y = y.to(self.device)
                            outputs = self.model(X)
                            loss = self.criterion(outputs.to(self.device), y)

                            training_loss += loss.item()
                            _, predicted = torch.max(outputs, 1)
                            training_correct += (predicted == y).sum().item()
                            training_samples += y.size(0)
                            global_steps += 1
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            log_every = 5

                            if (i + 1) % (
                                len(self.train_loader) // log_every
                            ) == 0 and len(training_stats) < log_every:
                                training_stats[i] = {
                                    "training_correct": training_correct,
                                    "training_accuracy": training_correct
                                    / training_samples,
                                    "training_loss": training_loss / training_samples,
                                    "hor_comm": self.communications_count["horizontal"],
                                    "ver_comm": self.communications_count["vertical"],
                                }

                        utilities.log_print(
                            f"holon({self.id}): run({self.global_run})-round({current_round})-epoch({e+1}), accuracy: {100*training_correct/training_samples :.3f}%"
                        )
                        epoch_stats[e + 1] = training_stats
                        round_stats[current_round].update(epoch_stats)

                        self.run_logs["train"].update(round_stats)

                        test_accuracy, test_loss = self.test(self.test_loader)
                        global_test_accuracy, global_test_loss = None, None
                        if self.has_global_test_data:
                            global_test_accuracy, global_test_loss = self.test(
                                self.global_test_data_loader
                            )
                        epoch_test_stats[e + 1] = {
                            "test_accuracy": test_accuracy,
                            "test_loss": test_loss,
                            "global_test_accuracy": global_test_accuracy,
                            "global_test_loss": global_test_loss,
                        }
                        round_test_stats[current_round].update(epoch_test_stats)
                        self.run_logs["test"].update(round_test_stats)

                    """communicating with neighbors"""
                    utilities.log_print(
                        f"holon({self.id}): communicating with neighbors"
                    )
                    for nid, nv in self.neighbors.copy().items():
                        # utilities.log_print(f"holon({self.id}): my neighbors are: {list(self.neighbors.keys())}")
                        my_state_dict = self.model.state_dict()
                        msg_type = (
                            "quit"
                            if self.global_run == runs and current_round == total_rounds
                            else "hint"
                        )
                        my_message = Message(msg_type, my_state_dict, msg_from=self.id)
                        my_message.msg_to = nid
                        if nid in list(self.neighbors.keys()):
                            try:  # in case the receiver has quitted already
                                self.send_msg(my_message.to_tensor(), nv[0].rank, nid)
                                # snd_reqs.append(dist.isend(my_message.to_tensor(), nv[0].rank))
                            except:
                                utilities.log_print(
                                    f"holon({self.id}): couldn't send a message to holon({nid})"
                                )
                                pass

                    current_round += 1
                    self.save_logs(str(self.global_run))

                """sending the trained model to the super holon"""
                if self.is_connected_to_superh:
                    utilities.log_print(
                        f"holon({self.id}): all set rounds finished, sending the trained model to the super holon"
                    )
                    my_state_dict = self.model.state_dict()
                    msg_type = (
                        "quit"
                        if self.global_run == runs and current_round > total_rounds
                        else "inform"
                    )
                    utilities.log_print(
                        f"holon({self.id}): message type is: {msg_type}"
                    )
                    my_message = Message(msg_type, my_state_dict, msg_from=self.id)

                    """let's send the model to the super and wait for any update from it"""
                    for k, u in self.superhs.items():

                        try:
                            my_message.msg_to = k
                            utilities.log_print(
                                f"holon({self.id}):sending to superholon ({u[0].id})"
                            )

                            self.send_msg(my_message.to_tensor(), u[0].rank, k)
                        except RuntimeError:
                            self.should_quit = True
                            self.quit_reason = "Couldn't connect to the super holon."
                            utilities.log_print(
                                f"holon({self.id}): couldn't connect to the super holon"
                            )

                else:
                    utilities.log_print(
                        f"holon({self.id}): all set rounds finished, but I am not connected to any super holon. I'll just wait for a signal from a neighbor except for the last run"
                    )
                    if self.global_run == runs and current_round > total_rounds:
                        pause_event.set()

                self.global_run += 1
                # self.should_pause = True
                utilities.log_print(
                    f"holon({self.id}): waiting for greenlight from the super"
                )
                pause_event.wait()
                utilities.log_print(
                    f"holon({self.id}): finished run({self.global_run-1})"
                )
                pause_event.clear()
                utilities.log_print(f"holon({self.id}): cleared the pause event")

        if (
            self.global_run > runs
        ):  # reached the run number limitations is the reason for quitting
            self.should_quit = True
            self.quit_reason = "reached max number of runs"
            utilities.log_print(
                f"holon({self.id}): runs and quit status are {runs} and {self.should_quit}"
            )
            quit_event.set()

        quit_event.wait()
        msgQ.put("finished")
        utilities.log_print(
            f"holon({self.id}): quitting because - {self.quit_reason} and its message threads live status are {rcv_thread.is_alive()} and {log_thread.is_alive()}"
        )
        rcv_thread.join()
        log_thread.join()
        utilities.log_print(f"holon({self.id}): joined the threads")
        print(f"holon({self.id}): finished the training process....")
        dist.destroy_process_group()
        # self.terminate()
        # return

    def test(self, test_data_loader):
        if not test_data_loader:
            raise Exception("No test dataset provided for testing.")
        n_samples = 0
        n_correct = 0
        loss = 0
        with torch.no_grad():
            for td, tt in test_data_loader:
                n_samples += tt.size(0)
                outputs = self.model(td)
                loss += self.criterion(outputs, tt).item()
                _, predicted = torch.max(outputs, 1)
                n_correct += (predicted == tt).sum().item()
        accuracy = n_correct / n_samples
        avg_loss = loss / n_samples
        return accuracy, avg_loss


class NonTerminalMLAvgHolon(NonTerminalMLHolon):
    def __init__(
        self,
        id,
        rank,  # for torch distribute process use
        neighbors,
        subhs,
        superhs,
        ml_model,
        starting_model=None,
        global_test_data=None,
        is_aggregator=True,
        device=None,
        log_dir="",
        sys_name="HL",
        synced=False,
        is_connected_to_superh=True,
        connected_subhs={},
        **kwargs,
    ) -> None:
        super().__init__(
            id=id,
            neighbors=neighbors,
            subhs=subhs,
            connected_subhs=connected_subhs,
            superhs=superhs,
            ml_model=ml_model,
            data=None,
            global_test_data=global_test_data,
            kwargs=kwargs,
            starting_model=starting_model,
            is_aggregator=is_aggregator,
            synced=synced,
            is_connected_to_superh=is_connected_to_superh,
        )
        self.rank = rank
        self.device = device
        self.received_neighbor_messages = dict()
        self.received_sub_messages = dict()
        self.received_sup_messages = dict()
        self.received_req_messages = dict()
        self.weight = self._calculate_self_weight()  # holon's self-importance weight
        self.message_size = Message(
            "dummy", self.model.state_dict(), msg_from=self.id
        ).get_tensor_size()
        self.should_quit = False
        self.quit_reason = "Unspecified"
        self.global_run = 0
        self.log_dir = os.path.join(log_dir, self.id)
        # os.makedirs(self.log_dir)
        self.sys_name = sys_name
        self.run_logs = {
            "train": {},
            "test": {},
            "system": self.sys_name,
            "data_dist": [],
            "communication": [{"horizontal": 0, "vertical": 0}],
        }
        self.should_pause = False
        self.sub_msg_count = 0
        self.is_aggregation_valid = False

    def set_log_dir(self, log_dir):
        self.log_dir = os.path.join(log_dir, self.id)
        # os.makedirs(self.log_dir)

    def set_sys_name(self, new_name):
        self.sys_name = new_name
        self.run_logs = {
            "train": {},
            "test": {},
            "system": self.sys_name,
            "data_dist": [],
            "communication": [{"horizontal": 0, "vertical": 0}],
        }

    def send_msg(self, msg_tensor, dst, dst_id, timeout=datetime.timedelta(seconds=30)):
        msg_size_tensor = torch.tensor([msg_tensor.size()[0]])
        if msg_size_tensor.item() >= Globals.max_message_tensor_size:
            raise Exception(
                f"holon({self.id}):the message will not fit in the tensor to be sent. Increase max_message_tensor_size"
            )
        else:
            zero_pad_tensor = torch.zeros(
                Globals.max_message_tensor_size - msg_size_tensor.item() - 1
            )
            augmented_msg_tensor = torch.concat(
                [msg_size_tensor, msg_tensor, zero_pad_tensor]
            )
            if dst_id in list(
                chain(
                    self.neighbors.keys(),
                    self.connected_subhs.keys(),
                    self.superhs.keys(),
                )
            ):
                req = dist.isend(augmented_msg_tensor, dst)
                # try:
                req.wait(timeout)
                utilities.log_print(
                    f"holon({self.id}): sent a message to holon({dst_id})"
                )
                # except:
                #     utilities.log_print(
                #         f"holon({self.id}): couldn't send a message to holon({dst_id})"
                #     )
                #     pass

    def recv_msg(self, src=None, timeout=datetime.timedelta(seconds=1800)):
        whole_msg = torch.zeros(Globals.max_message_tensor_size)
        # dist.recv(whole_msg, src)
        req = dist.irecv(whole_msg, src)
        utilities.log_print(f"holon({self.id}): waiting for a message from {src}")
        req.wait(timeout)
        utilities.log_print(f"holon({self.id}): received a message from {src}")
        actual_msg_size = int(whole_msg[0].item())
        actual_msg_tensor = whole_msg[1 : actual_msg_size + 1].to(
            torch.ByteTensor().dtype
        )  # because the message initially serialiez into byte tensor
        return actual_msg_tensor

    def listen_to_messages(self, pause_event, quit_event):
        # utilities.log_print(f"holon({self.id}): messag size={self.message_size}")
        """An idea for the future. For now I am using pytorchs distributed library"""

        rcevd_msg_queue = Queue()
        process_msg_event = Event()
        process_msg_thread = threading.Thread(
            target=self.process_messages,
            args=(rcevd_msg_queue, pause_event, quit_event, process_msg_event),
        )
        process_msg_thread.start()
        while not quit_event.is_set():
            try:
                rcvd_msg = self.recv_msg()

                rcevd_msg_queue.put(rcvd_msg)
                process_msg_event.set()

            except RuntimeError:
                self.should_quit = True
                self.quit_reason = "Not received any meassages for a while"
                rcevd_msg_queue.put(None)
                process_msg_event.set()
                process_msg_thread.join()
                quit_event.set()
                # break
        rcevd_msg_queue.put(None)
        process_msg_event.set()
        process_msg_thread.join()
        quit_event.set()

    def log_stats(self, interval):
        """Logs the stats of the holon in the given interval
        for now it only logs the number of communications"""
        time_step = 0
        test_stats = {}
        while not self.should_quit:
            self.run_logs["communication"].append(self.communications_count)
            global_test_accuracy = self.test(self.global_test_data_loader)
            test_stats[1] = {
                "test_accuracy": None,
                "test_loss": None,
                "global_test_accuracy": global_test_accuracy,
                "global_test_loss": None,
            }
            self.run_logs["test"][time_step + 1] = test_stats
            self.save_logs(str(self.global_run))
            time_step += 1
            time.sleep(interval)
        print(f"holon({self.id}): loged {time_step} times")
        exit("holon({self.id}): log_stats thread exited")

    def _calculate_self_weight(self):
        sw = 0
        for (
            s
        ) in (
            self.subhs.values()
        ):  # the weight includes all subs regardless of having direct connection or not
            sw += s[0].weight
        return sw

    def _calculate_class_counts(self):
        if len(self.subhs) > 0:
            self.class_labels = next(iter(self.subhs.values()))[0].class_labels
            self.class_counts = [0] * len(self.class_labels)
            for s in self.subhs.values():
                self.class_counts = [
                    sum(x) for x in zip(self.class_counts, s[0].class_counts)
                ]
            return self.class_counts

    def add_subh(self, sub, w):
        super().add_subh(sub, w)
        self.weight = self._calculate_self_weight()

    def process_messages(
        self, rcevd_msg_queue, pause_event, quit_event, process_msg_event
    ):
        while True:
            # process_msg_event.wait()
            process_msg_event.clear()
            recved_msg = rcevd_msg_queue.get()
            if recved_msg is None:
                break
            self._process_messages(recved_msg, pause_event, quit_event)

    def _process_messages(self, message_tensor, pause_event, quit_event):
        message = Message.from_tensor(message_tensor)
        sender_id = message["msg_from"]
        utilities.log_print(
            f"holon({self.id}): received a ({message['msg_type']}) message from holon({sender_id})"
        )
        if message["msg_type"] == "inform":
            utilities.log_print(
                f"holon({self.id}): the connected subholons are {list(self.connected_subhs.keys())}"
            )
        if (
            message["msg_type"] == "hint"
            and sender_id in list(self.neighbors.keys())
            and set(self.superhs.keys())
            and set(self.neighbors[sender_id][0].superhs.keys())
        ):
            self.received_neighbor_messages[sender_id] = (
                message["msg_content"],
                self.neighbors[sender_id][1],
            )
            self.communications_count["horizontal"] += 1
        elif message["msg_type"] == "update":
            self.received_sup_messages[sender_id] = (
                message["msg_content"],
                self.superhs[sender_id][1],
            )
            self.should_pause = False
            pause_event.set()
        elif message["msg_type"] == "continue":
            self.should_pause = False
            self.should_quit = True
            pause_event.set()
            quit_event.set()  # because continue is only for quitting requests in superholons
        elif message["msg_type"] == "request":
            who_tuple = (
                self.neighbors[sender_id]
                if sender_id in list(self.neighbors.keys())
                else self.superhs[sender_id]
            )
            self.received_req_messages[sender_id] = (
                message["msg_content"],
                who_tuple[1],
            )
            self.should_pause = False
            pause_event.set()
        elif message["msg_type"] == "inform" and sender_id in list(
            self.connected_subhs.keys()
        ):
            if self.is_aggregator:
                utilities.log_print(
                    f"holon({self.id}): adding the received sub model to database"
                )
                self.received_sub_messages[sender_id] = (
                    message["msg_content"],
                    self.connected_subhs[sender_id][1],
                )
                self.communications_count["vertical"] += 1
            else:
                utilities.log_print(
                    f"holon({self.id}): trying to send continue message to holon({sender_id})"
                )
                self.sub_msg_count += 1

                if self.synced:
                    utilities.log_print(
                        f"holon({self.id}): sub_msg_count={self.sub_msg_count} and len(self.connected_subhs)={len(self.connected_subhs)}"
                    )
                    if self.sub_msg_count == len(
                        self.connected_subhs
                    ):  # synchronizing the subholons
                        for sid in self.connected_subhs.keys():
                            c_message = Message(
                                "continue",
                                self.model.state_dict(),
                                msg_from=self.id,
                                msg_to=sid,
                            )
                            self.send_msg(
                                c_message.to_tensor(),
                                self.connected_subhs[sid][0].rank,
                                sid,
                            )
                        self.sub_msg_count = 0
                else:
                    c_message = Message(
                        "continue",
                        self.model.state_dict(),
                        msg_from=self.id,
                        msg_to=sender_id,
                    )
                    self.send_msg(
                        c_message.to_tensor(),
                        self.connected_subhs[sender_id][0].rank,
                        sender_id,
                    )

            self.should_pause = False
            pause_event.set()
        elif message["msg_type"] == "quit":
            if sender_id in list(self.connected_subhs.keys()):
                utilities.log_print(
                    f"holon({self.id}): the quit message is from a subholon"
                )
                # if len (set(self.subhs.keys()) - {sender_id})>0:
                # """this is not the last subholon to quit so let's store it aggregate"""
                self.received_sub_messages[sender_id] = (
                    message["msg_content"],
                    self.connected_subhs[sender_id][1],
                )  # the last message from the subholon containing its model state dict
                c_message = Message(
                    "continue",
                    self.model.state_dict(),
                    msg_from=self.id,
                    msg_to=sender_id,
                )
                utilities.log_print(
                    f"holon({self.id}): sending a (continue) message to holon({sender_id})"
                )
                self.send_msg(
                    c_message.to_tensor(),
                    self.connected_subhs[sender_id][0].rank,
                    sender_id,
                )  # so that the subholon can continue quitting
                self.remove_subh(sender_id)
                utilities.log_print(
                    f"holon({self.id}): the subholons after removing {sender_id} are {list(self.connected_subhs.keys())}"
                )
                if len(self.connected_subhs) == 0:
                    utilities.log_print(
                        f"holon({self.id}): all subholons quitted. starting my own quit process"
                    )
                    self.aggregate_models(
                        "all"
                    )  # preparing the last model aggregation before quitting
                    if len(self.neighbors) > 0:
                        for ni, nv in self.neighbors.items():
                            q_message = Message(
                                "quit",
                                self.model.state_dict(),
                                msg_from=self.id,
                                msg_to=ni,
                            )
                            utilities.log_print(
                                f"holon({self.id}): sending a (quit) message to holon({ni})"
                            )
                            self.send_msg(q_message.to_tensor(), nv[0].rank, ni)
                    if len(self.superhs) > 0 and self.is_connected_to_superh:
                        my_state_dict = self.model.state_dict()
                        for supid, sup in self.superhs.items():
                            q_message = Message(
                                "quit", my_state_dict, msg_from=self.id, msg_to=supid
                            )
                            utilities.log_print(
                                f"holon({self.id}): sending a (quit) message to holon({supid})"
                            )
                            self.send_msg(q_message.to_tensor(), sup[0].rank, supid)
                            # dist.send(q_message.to_tensor(), sup[0].rank)
                    else:
                        utilities.log_print(
                            f"holon({self.id}): I am the root holon, so I will quit"
                        )
                        self.should_quit = True
                        quit_event.set()
                        pause_event.set()
                    self.quit_reason = "all subholons quitted"
                else:
                    utilities.log_print(
                        f"holon({self.id}): still have subholons {list(self.connected_subhs.keys())} alive"
                    )
                self.communications_count["vertical"] += 1
                # pause_event.set()
            elif sender_id in list(self.neighbors.keys()):
                utilities.log_print(
                    f"holon({self.id}): the quit message is from a neighbor"
                )
                self.received_neighbor_messages[sender_id] = (
                    message["msg_content"],
                    self.neighbors[sender_id][1],
                )  # the last message from the subholon containing its model state dict
                self.remove_neighbor(self.neighbors[sender_id][0])
                utilities.log_print(
                    f"holon({self.id}): the neighbors after removing {sender_id} are {list(self.neighbors.keys())}"
                )
                self.communications_count["horizontal"] += 1
            elif sender_id in list(self.superhs.keys()):
                # TODO: what to do if superholon terminates?
                raise Exception("Superholon terminates first: Not implemented yet")

    def aggregate_models(self, which_models):
        s_dicts = []
        s_ws = []
        w_offset = 0
        if which_models == "neighbors":
            for m in self.received_neighbor_messages.values():
                s_dicts.append(m[0])
                s_ws.append(m[1])
                w_offset = self.weight if self.has_aggregated else 0
        elif which_models == "subs":
            for m in self.received_sub_messages.values():
                s_dicts.append(m[0])
                s_ws.append(m[1])
        elif which_models == "supers":
            for m in self.received_sup_messages.values():
                s_dicts.append(m[0])
                s_ws.append(m[1])
                # w_offset = self.weight if self.has_aggregated else 0  # experimenting
        elif which_models == "all":
            if len(self.received_sup_messages) > 0:
                self.is_aggregation_valid = True
                for m in self.received_sup_messages.values():
                    s_dicts.append(m[0])
                    s_ws.append(m[1])
            else:
                if self.received_enough_hints():
                    self.is_aggregation_valid = True
                else:
                    self.is_aggregation_valid = False
                for m in self.received_neighbor_messages.values():
                    s_dicts.append(m[0])
                    s_ws.append(m[1])
                for m in self.received_sub_messages.values():
                    s_dicts.append(m[0])
                    s_ws.append(m[1])

        else:
            raise Exception(f"Unknown aggregation models {which_models}")

        if len(s_dicts) == 0:
            return

        agg_state_dict = copy.deepcopy(self.model.state_dict())
        for pk, pv in agg_state_dict.items():
            if w_offset == 0:
                agg_state_dict[pk] = torch.zeros(pv.size())  # initialized with 0
            else:
                agg_state_dict[pk] = torch.mul(
                    w_offset, pv
                )  # initialized with my tensor. can support the above statement too, but added for clarity
            for i, s in enumerate(s_dicts):
                agg_state_dict[pk] += torch.mul(s_ws[i], s[pk]).to(
                    torch.device("cpu")
                    if agg_state_dict[pk].get_device() == -1
                    else self.device
                )
            agg_state_dict[pk] /= w_offset + sum(s_ws)
        # self.model.load_state_dict({key: value.to(torch.device("cpu") if self.model.get_device()==-1 else self.device) for key, value in agg_state_dict.items()})
        model_dev = next(self.model.parameters()).device
        self.model.load_state_dict(
            {key: value.to(model_dev) for key, value in agg_state_dict.items()}
        )
        utilities.log_print(
            f"holon({self.id}): finished the aggregation of {which_models}. The weights were {s_ws}, and the offset was {w_offset} and my weight is {self.weight}"
        )
        self.has_aggregated = True

    def received_enough_informs(self):
        for k, v in self.connected_subhs.copy().items():
            if k not in list(
                self.received_sub_messages.keys()
            ) and k in list(  # or k not in list(
                self.connected_subhs.keys()
            ):
                utilities.log_print(
                    f"holon({self.id}): received_enough_inform: False because not received a message from subholon({k}). The received messages are {list(self.received_sub_messages.keys())} and the connected subholons are {list(self.connected_subhs.keys())}"
                )
                return False
        return len(self.received_sub_messages) > 0

    def received_enough_hints(self):
        insiders_num = 0
        for k, v in self.neighbors.copy().items():
            if set(self.superhs.keys()) and set(
                self.neighbors[k][0].superhs.keys()
            ):  # the neighbor belongs to the same superholon
                insiders_num += 1
                if k not in list(self.received_neighbor_messages.keys()) and k in list(
                    self.neighbors.keys()
                ):
                    return False
        return len(self.received_neighbor_messages) > 0 or insiders_num == 0

    def received_enough_updates(self):
        for k, v in self.superhs.copy().items():
            if k not in list(self.received_sup_messages.keys()) or k not in list(
                self.superhs.keys()
            ):
                return False
        return len(self.received_sup_messages) > 0

    def save_logs(self, run="0"):
        utilities.save_dict_to_pickle(self.run_logs, self.log_dir, f"run_{run}")

    def terminate(self):
        self.save_logs(str(self.global_run))

    def train(self, msgQ, rounds, runs, max_message_tensor_size):
        Globals.max_message_tensor_size = max_message_tensor_size
        print(f"holon({self.id}): starting the training process....")
        utilities.log_print(
            f"holon({self.id}): connected subholons are {list(self.connected_subhs.keys())}, subholons are {list(self.subhs.keys())} and is_connected_to_superh={self.is_connected_to_superh}"
        )
        os.makedirs(self.log_dir)
        pause_event = threading.Event()
        quit_event = threading.Event()
        rcv_thread = threading.Thread(
            target=self.listen_to_messages, args=(pause_event, quit_event)
        )
        log_thread = threading.Thread(target=self.log_stats, args=(1,))
        rcv_thread.start()
        log_thread.start()
        training_loss = 0
        training_correct = 0
        training_samples = 0
        global_steps = 0
        total_rounds = rounds
        while not self.should_quit and len(self.connected_subhs) > 0:
            current_round = 1
            round_stats = {}
            while current_round <= total_rounds + 1:
                hint_neighbors = True
                inform_super = (
                    current_round == total_rounds + 1 and self.is_connected_to_superh
                )  # inform the super only immidiately after my designated total rounds
                update_subs = (
                    current_round <= total_rounds
                    or len(self.superhs) == 0
                    or not self.is_connected_to_superh
                )  # update the subs only if there is no super or the current round is less than the total rounds
                if self.received_enough_informs():
                    utilities.log_print(
                        f"holon({self.id}): received enough informs and current connected subs are {list(self.connected_subhs.keys())}"
                    )
                    # self.aggregate_models("subs")
                    if self.received_enough_updates():
                        self.aggregate_models("supers")
                        hint_neighbors = False
                    else:
                        self.aggregate_models("subs")
                        subs_agg_state_dict = copy.deepcopy(
                            self.model.state_dict()
                        )  # superholon will be informed about the subs aggregation only
                        """hinting the neighbors about the last aggregation"""
                        if (
                            self.is_aggregator
                            and hint_neighbors
                            and len(self.neighbors) > 0
                        ):
                            utilities.log_print(
                                f"holon({self.id}): communicating with neighbors"
                            )
                            for nid, nv in self.neighbors.copy().items():
                                if nid in list(self.neighbors.keys()):
                                    utilities.log_print(
                                        f"holon({self.id}): neighbor {nid} is alive, so let's send a hint"
                                    )
                                    my_state_dict = self.model.state_dict()
                                    my_message = Message(
                                        "hint",
                                        my_state_dict,
                                        msg_from=self.id,
                                        msg_to=nid,
                                    )
                                    utilities.log_print(
                                        f"holon({self.id}): sending (hint) message to {nid}"
                                    )
                                    self.send_msg(
                                        my_message.to_tensor(), nv[0].rank, nid
                                    )

                        """ wait for the neighbors to send their hints"""
                        while (not self.received_enough_hints()) and self.synced:
                            time.sleep(0.1)

                        self.aggregate_models("neighbors")

                    """reseting the received messages for the next collaboration round. Old messages are not needed anymore"""
                    self.received_sub_messages = dict()
                    self.received_neighbor_messages = dict()
                    self.received_sup_messages = dict()
                    utilities.log_print(
                        f"holon({self.id}): is_aggregator={self.is_aggregator} and current_round={current_round} and total_rounds={total_rounds}"
                    )
                    if self.is_aggregator and update_subs:
                        for i, sv in enumerate(
                            self.connected_subhs.copy().values()
                        ):  # for selective sub hints, this part should be changed
                            # dist.send(my_state_tensor, sv[0].rank)
                            if sv[0].id in list(
                                self.connected_subhs.keys()
                            ):  # to make sure that the sub has not quit during the aggregation process
                                utilities.log_print(
                                    f"holon({self.id}): sending the aggregated results to holon({sv[0].id})"
                                )
                                my_state_dict = self.model.state_dict()
                                my_message = Message(
                                    "update",
                                    my_state_dict,
                                    msg_from=self.id,
                                    msg_to=sv[0].id,
                                )
                                # my_message.msg_to = sv[0].id
                                try:  # in case the receiver is quitted already
                                    utilities.log_print(
                                        f"holon({self.id}): sending (update) message to {sv[0].id}"
                                    )
                                    self.send_msg(
                                        my_message.to_tensor(), sv[0].rank, sv[0].id
                                    )
                                    # dist.send(my_message.to_tensor(), sv[0].rank)
                                except:
                                    pass

                    """communicating with supers about the last aggregation"""
                    utilities.log_print(
                        f"holon({self.id}): is_aggregator={self.is_aggregator} and inform_super={inform_super} and len(self.superhs)={len(self.superhs)} and is_connected_to_superh={self.is_connected_to_superh}"
                    )
                    if (
                        inform_super
                        and len(self.superhs) > 0
                        and self.is_connected_to_superh
                    ):
                        utilities.log_print(
                            f"holon({self.id}): communicating with supers about the last aggregation"
                        )
                        for sid, sv in self.superhs.copy().items():
                            if sid in list(self.superhs.keys()):
                                # my_state_dict = self.model.state_dict()
                                my_state_dict = subs_agg_state_dict  # experimenting
                                my_message = Message(
                                    "inform",
                                    my_state_dict,
                                    msg_from=self.id,
                                    msg_to=sid,
                                )
                                utilities.log_print(
                                    f"holon({self.id}): sending (inform) message to {sid}"
                                )
                                self.send_msg(my_message.to_tensor(), sv[0].rank, sid)
                                # snd_reqs.append(dist.isend(my_message.to_tensor(), sv[0].rank))

                    round_stats[current_round] = (
                        {}
                    )  # TODO: should be completed appropriately

                    current_round += 1

                if (
                    self.received_enough_updates()
                    and not self.received_enough_informs()
                ):  # this is for the case that subs are waiting for update propgated from the super
                    utilities.log_print(
                        f"holon({self.id}): received enough updates at run {self.global_run} round {current_round}"
                    )
                    # self.aggregate_models("supers")
                    self.aggregate_models("supers")
                    """Forwarding to the subs"""
                    for nid, nv in self.connected_subhs.items():
                        my_state_dict = self.model.state_dict()
                        my_message = Message(
                            "update", my_state_dict, msg_from=self.id, msg_to=nid
                        )
                        utilities.log_print(
                            f"holon({self.id}): sending (update) message to {nid}"
                        )
                        self.send_msg(my_message.to_tensor(), nv[0].rank, nid)
                        # dist.send(my_message.to_tensor(), nv[0].rank)
                    self.received_sup_messages = dict()

                pause_event.wait()
                pause_event.clear()

                if len(self.connected_subhs) == 0:  # all subs quited
                    utilities.log_print(
                        f"holon({self.id}): just found all connected subs quitted"
                    )
                    if (
                        not self.is_connected_to_superh
                    ):  # if I am not connected to a superholon, then I should quit myself
                        quit_event.set()
                    break

            self.run_logs["train"].update({self.global_run: round_stats})
            self.global_run += 1
            utilities.log_print(
                f"holon({self.id}): finished run {self.global_run -1}. The quit status is {self.should_quit} and the connected subholons are {list(self.connected_subhs.keys())}"
            )
            if quit_event.is_set():
                break
        quit_event.wait()
        utilities.log_print(f"holon({self.id}): super gave a greenlight to quit")
        msgQ.put("finished")
        utilities.log_print(
            f"holon({self.id}): quitting because - {self.quit_reason} and its message threads live status are {rcv_thread.is_alive()} and {log_thread.is_alive()}"
        )
        rcv_thread.join()
        log_thread.join()
        utilities.log_print(f"holon({self.id}): joined the threads")
        print(f"holon({self.id}): finished the training process....")
        dist.destroy_process_group()
        # self.terminate()

    def test(self, test_data_loader):
        if not test_data_loader:
            raise Exception("No test dataset provided for testing.")
        n_samples = 0
        n_correct = 0
        self.model.to(self.device)
        with torch.no_grad():
            for td, tt in test_data_loader:
                td = td.to(self.device)
                tt = tt.to(self.device)
                n_samples += tt.size(0)
                outputs = self.model(td)
                _, predicted = torch.max(outputs, 1)
                n_correct += (predicted == tt).sum().item()
        accuracy = n_correct / n_samples
        return accuracy
