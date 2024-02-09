""" Defines the settings for the algorithms and experiments. """

class Globals(object):
    current_num_terminal_holons = 0
    current_num_nonterminal_holons = 0
    max_message_tensor_size = 0
    DEBUG_MODE = False
    
    @classmethod
    def init(cls):
        cls.current_num_terminal_holons = 0
        cls.current_num_nonterminal_holons = 0