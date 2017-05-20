"""
This is a project by Satyaki Sanyal.
This project must be used for educational purposes only.
Follow me on:
LinkedIn - https://www.linkedin.com/in/satyaki-sanyal-708424b7/
Github - https://github.com/Satyaki0924/
Researchgate - https://www.researchgate.net/profile/Satyaki_Sanyal
"""

# ***** DO NOT CHANGE ANYTHING IN THIS FILE *****

from src.plot import Plot

def main():
    while True:
        try:
            ip = int(input('Enter 1. to analyse tweets, 2. to visualise tweets, 3. Exit \n>> '))
            if ip == 1:
                from src import twitter
            elif ip == 2:
                Plot.plot()
            elif ip == 3:
                print('*** Thank you ***')
                break
            else:
                print('*** Input not recognized. Try Again! ***')
        except Exception as e:
            print('***** EXCEPTION FACED: ' + str(e) + ' *****')

if __name__ == '__main__':
    main()

# ***** END OF FILE *****
