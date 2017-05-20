# Sentiment Prediction and graphing of live tweeter data with recurrent nets and lstm
This project analyses live tweeter sentiments and visualises them using recurrent neural networks and long short term memories. I have used recurrent nets because while training on huge data, recurrent nets actually predicts the outcome a lot better than any normal machine learning models.

##### *** This project may throw errors if trained on CPU instead of GPU ***

### This project is configured for Linux and uses python3
To run this project, open up your bash terminal and write

```
chmod -R 777 setup.sh
./setup.sh
```

This will set up the project enviornment for you. This must be run with administrator rights.
After you set up the project, run:

#### Setup Virtual enviornment

```
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Train and test the accuracy of the project

```
python configure.py
```

#### Analyse and visualise twitter data

```
python run_me.py

```

#### keyword = Ransomware
![Terminal screen_4](https://github.com/Satyaki0924/sentiment-prediction-and-graphing-of-live-tweeter-data-with-recurrent-nets-and-lstm/blob/master/res/tweet.png?raw=true "Terminal4")

### Plots:

#### keyword = Modi
![Terminal screen_1](https://github.com/Satyaki0924/sentiment-prediction-and-graphing-of-live-tweeter-data-with-recurrent-nets-and-lstm/blob/master/res/modi-plot.png?raw=true "Terminal1")

#### keyword = Trump
![Terminal screen_2](https://github.com/Satyaki0924/sentiment-prediction-and-graphing-of-live-tweeter-data-with-recurrent-nets-and-lstm/blob/master/res/trump-plot.png?raw=true "Terminal2")

#### keyword = Ransomware
![Terminal screen_3](https://github.com/Satyaki0924/sentiment-prediction-and-graphing-of-live-tweeter-data-with-recurrent-nets-and-lstm/blob/master/res/ransomware-plot.png?raw=true "Terminal3")


#### Author: Satyaki Sanyal
*** This project is strictly for educational purposes only. ***
