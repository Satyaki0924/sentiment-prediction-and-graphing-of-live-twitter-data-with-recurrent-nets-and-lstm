"""
This is a project by Satyaki Sanyal.
This project must be used for educational purposes only.
Follow me on:
LinkedIn - https://www.linkedin.com/in/satyaki-sanyal-708424b7/
Github - https://github.com/Satyaki0924/
Researchgate - https://www.researchgate.net/profile/Satyaki_Sanyal
"""
from src.main import Main


def main():
    lstm_size = 256
    lstm_layers = 1
    batch_size = 500
    learning_rate = 0.001
    while True:
        try:
            ip = int(input('Enter 1. to train, 2. to test accuracy, 3. to test manually, 4. Exit \n>>'))
            if ip == 1 or ip == 2 or ip == 3:
                Main(lstm_size, lstm_layers, batch_size, learning_rate).execute(ip)
            elif ip == 4:
                print('*** Thank you ***')
                break
            else:
                print('*** Input not recognized. Try Again! ***')
        except Exception as e:
            print('***** EXCEPTION FACED: ' + str(e) + ' *****')

    if __name__ == '__main__':
        main()
