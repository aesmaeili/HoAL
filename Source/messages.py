"""Defining the structure of the messages exchanged between holons"""
import pickle
import torch


class Message(object):
    def __init__(self, msg_type, msg_content, msg_from="unknown", msg_to="unknown") -> None:
        self.msg_type = msg_type
        self.msg_content = msg_content
        self.msg_from = msg_from
        self.msg_to = msg_to

    def to_tensor(self):
        pic = pickle.dumps(self.__dict__)
        tensor = torch.ByteTensor(bytearray(pic))
        return tensor
    
    def get_tensor_size(self):
        return self.to_tensor().size()
    
    def to_dict(self):
        return self.__dict__
    
    def __getitem__(self, key):
        return self.__dict__[key]
    
    @classmethod
    def from_tensor(cls, a_tensor):
        bytess = a_tensor.numpy().tobytes()
        a_dict = pickle.loads(bytess)
        return cls(**a_dict)
    
if __name__ == "__main__":
    m = Message("t", "co", "me", "him")
    t=m.to_tensor()
    print(m.get_tensor_size())
    t2 = Message.from_tensor(t)
    print(t2.to_dict())