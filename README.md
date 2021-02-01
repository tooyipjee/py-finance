# Personal Finance
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/L3L33I8CT)

Personal finance is an analysis tool that performs various time-series analysis on stock data, finds the optimum portfolio of a group of stocks and generate forecast share prices.  

* Tool to download **real-time** share price from Yahoo Finance and Investing.com.
* **Time-series analysis** on share price data. Butterworth filtering to smooth the prices and complete incomplete data that the sources above can sometimes be introduced with data from emerging markets, and Fast Fourier Transforms (FFT) to identify any dominant periodicity in the share price.
* **Portfolio** tools that calculate volatility, returns and the optimum partfolio. It also helps translate those insights into actionable investing decisions.  
* **Forecast** share price data using three generative Tensorflow models (GAN) such as the Long Short Term Memory (LSTM), Gated Recurrent Unit (GRU) and the Dilated Convolutional Neural Network (DCNN). The idea behind this techniques is to try to capture a representation of the underlying latent pocess of the time-series data and generate synthetic data from the date of the most recent data point. 

## How It Works

1. Personal finance contains a Portfolio class that wraps over two pre-existing libraries, `PyPortfolioOpt`<a id="1">[1]</a>  and `Stock Prediction Models`<a id="1">[2]</a> . 
2. The Portfolio class string input which should point to a CSV file that contains the symbols of the stocks you want to analyse.
3. The `download stocks()` function downloads the stock price for the specified time horizon and saves the data in the `Portfolio()` object.  
4. The `Portfolio` object contains methods that will perform the analysis on the saved stocks which is why the `download_stocks()` function should be run, followed by `create_portfolio()`.
5. Generation of synthetic forecast data can also be performed on the symbols and defaults to using the whole period of downloaded stock price to train the generative models. 


## Installation

The installation of this tool can be done using the `anaconda`, https://www.anaconda.com/products/individual. As the code uses `cvxpy`, do ensure that you the latest version of the Visual Studio C++ build tool, you can download them here, https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16. The version tested with is, `MSVC v142 - VS 2019 C++ x64/x86 build tools (v14.28)`.

1. Clone the repository using the command below.

    ```sh
    $ git clone https://github.com/tooyipjee/personal-finance.git
    ```

2. Navigate to the `personal-finance` directory and create the environment which installs all the necessary libraries by running the following command.

    ```sh
    $ conda env create -f environment.yml
    ```
    
3. Activate the environment.

    ```sh
    $ conda activate py-finance
    ```
    
3. Run the Spyder IDE, or the IDE of your choice. In the command line interface, run the following.

    ```sh
    $ spyder
    ```
    
5. Edit this line of `main.py` to point towards the `personal-finance/` directory. If using default Github settings, it be as easy as replacing the ###### to your User Account Folder.

    ```py
    ...
    $ wd = "C:/Users/######/Documents/GitHub/personal-finance/"
    ...
    ```
    
5. Run `main.py` by clicking the run button in Spyder or run the following command in the environemnt.

    ```sh
    $ python main.py
    ```
## Features
### Downloading Prices<a id="1">[6]</a> 
![alt text](https://github.com/tooyipjee/personal-finance/blob/master/reference/images/MBMR_price.png)
Downloaded and filtered prices from Yahoo Finance and Investing.com with the Index trend.
### Optimum Portfolio<a id="1">[5]</a> 
![alt text](https://github.com/tooyipjee/personal-finance/blob/master/reference/images/efficient_frontier.png)
Efficient frontier.
![alt text](https://github.com/tooyipjee/personal-finance/blob/master/reference/images/weights_bar.png)
Distribution of stocks for optimum portfolio.
![alt text](https://github.com/tooyipjee/personal-finance/blob/master/reference/images/which_stock.png)
Volatility-return scatter plot of symbols.
### Fast Fourier Transform<a id="1">[3]</a> 
![alt text](https://github.com/tooyipjee/personal-finance/blob/master/reference/images/MBMR_fft.png)
Periodicity of share price data. High peaks mean that there is a periodic "signal" with a period the inverse of the the value in the x-axis in days. (1/x-axis value days)
### Monte Carlo Siumlation
![alt text](https://github.com/tooyipjee/personal-finance/blob/master/reference/images/MBMR_mc_drift.png)
Monte Carlo simulation of stock returns for the next __ days.
### Forecast<a id="1">[4]</a> 
![alt text](https://github.com/tooyipjee/personal-finance/blob/master/reference/images/forecast_LSTM_MBMR.png)
Forecasted share price.


## Bugs and feedback

I am stil actively maintaining this repository.
Please raise an issue to report any bugs and suggest improvements. 

## References
<a id="1">[1]</a> 
Husein Zolkelpi (2020). 
Stock-Prediction-Models (https://github.com/huseinzol05/Stock-Prediction-Models)

<a id="1">[2]</a> 
Robert Andrew Martin  (2020). 
PyPortfolioOpt (https://github.com/robertmartin8/PyPortfolioOpt)

<a id="1">[3]</a> 
Bruce G. Lewisa
, Ric D. Herbertb and Rod D. Bell (2006). 
The Application of Fourier Analysis to Forecasting the
Inbound Call Time Series of a Call Centre (https://www.mssanz.org.au/MODSIM03/Volume_03/B10/06_Lewis.pdf)

<a id="1">[4]</a> 
Ricardo Alberto Carrillo Romero
 (2019). 
Generative Adversarial Network for Stock Market
price Prediction (https://cs230.stanford.edu/projects_fall_2019/reports/26259829.pdf)

<a id="1">[5]</a> 
Harry Markowitz
 (1952). 
Portfolio Section (https://www.math.ust.hk/~maykwok/courses/ma362/07F/markowitz_JF.pdf)

<a id="1">[6]</a> 
S. Butterworth
 (1930). 
On the Theory of Filter Amplifiers(https://www.changpuak.ch/electronics/downloads/On_the_Theory_of_Filter_Amplifiers.pdf)
