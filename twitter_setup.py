"""
***** SETUP YOUR TOKEN AND ACCESS CODES IN THE RESPECTIVE PLACES *****
"""


class Setup(object):
    @staticmethod
    def get():
        # *** consumer key, consumer secret, access token, access secret. ***
        c_key = "ajksnadjksnfkjsdnjk3iep239ro24jiorj24oij4opi234j2r43rdf"
        c_secret = "ajksnadjksnfkjsdnjk3iep239ro24jiorj24oij4opi234j2r43rdf"
        acc_token = "ajksnadjksnfkjsdnjk3iep239ro24jiorj24oij4opi234j2r43rdf"
        acc_secret = "ajksnadjksnfkjsdnjk3iep239ro24jiorj24oij4opi234j2r43rdf"
        return c_key, c_secret, acc_token, acc_secret

    @staticmethod
    def get_val():
        lstm_size = 256
        lstm_layers = 1
        batch_size = 1
        learning_rate = 0.001
        return lstm_size, lstm_layers, batch_size, learning_rate
