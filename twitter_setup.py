"""
***** SETUP YOUR TOKEN AND ACCESS CODES IN THE RESPECTIVE PLACES *****
"""


class Setup(object):
    @staticmethod
    def get():
        # *** consumer key, consumer secret, access token, access secret. ***
        c_key = "5FasqkkqsrwaLTKR4LQwjnuFT"
        c_secret = "ppsjkcWvIF3nBDYlE6ZxMOQ6KwLJWtnka1Nr63yL1CxQ4q6ucS"
        acc_token = "702550498524557312-MrIGfQy25GHQfmLwx6oNBK8pvt52JWN"
        acc_secret = "L1OTnMPed3OJnkWfse9cepzVjBWg7Op50YVjBg3UeqtGh"
        return c_key, c_secret, acc_token, acc_secret

    @staticmethod
    def get_val():
        lstm_size = 256
        lstm_layers = 1
        batch_size = 1
        learning_rate = 0.001
        return lstm_size, lstm_layers, batch_size, learning_rate
