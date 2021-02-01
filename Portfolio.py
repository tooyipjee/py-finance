import pandas as pd
import numpy as np
import cvxpy as cp
import os
import yfinance as yf
import datetime as dt
import csv
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import EfficientFrontier
from pypfopt import objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import HRPOpt
from pypfopt import CLA
from pypfopt import black_litterman
from pypfopt import BlackLittermanModel
from pypfopt import plotting
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import investpy
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter('ignore')
import tensorflow.compat.v1 as tf
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates
import matplotlib.mlab as mlab
import scipy.signal as dsp
from scipy.stats import norm
from scipy.fft import fft, fftfreq
import copy

import models.lstm as lstm
import models.dcnn as dcnn
import models.gru2 as gru2

class Portfolio:
    stock_prices = pd.DataFrame()
    details = [] 
    save_output = False
    output_path = ""
    tf.disable_v2_behavior()
    
    def create_portfolio(self, duration, holding_path, output_path):
        self.today = dt.date.today() - dt.timedelta(days = 1)
        self.start = self.today - dt.timedelta(days = duration * 365)
        self.holding_name = holding_path
        self.holding = pd.read_csv(holding_path)
        self.name = (holding_path.split('symbols/')[1]).split('.')[0]    
        self.output_path = output_path
        self.forecast_stock_prices = pd.DataFrame()
        self.index_price = pd.DataFrame()
        self.lstm_forecast_price = pd.DataFrame()
        self.dcnn_forecast_price = pd.DataFrame()
        self.gru2_forecast_price = pd.DataFrame()
        self.cash = 1000000
        
    def download_stocks(self, source, country):
        for i, symbol in enumerate(self.holding.Symbol):
            download_success = True
            if source == "yahoo":
                data = yf.download(symbol, start=self.start, end=self.today)
                
            elif source == "investing":
                try:
                    data = investpy.get_stock_historical_data(
                        stock=symbol.split('.')[0],
                        country=country,
                        from_date=str(self.start.day)+'/'+str(self.start.month)+'/'+str(self.start.year),
                        to_date=str(self.today.day)+'/'+str(self.today.month)+'/'+str(self.today.year))
                except:
                    print(symbol.split('.')[0] +  " not found!")
                    download_success = False
            if download_success:            # days with bad data
                bad_days = data[data.Close == 0].index
                if data.shape[0] >= 2:
                    print(data.shape)
                    for bad_day in bad_days:
                        avg_close_price = (data.loc[bad_day - dt.timedelta(days = 5):bad_day + dt.timedelta(days = 5)].Close)
                        avg_close_price = np.mean(avg_close_price)               
                        data.at[bad_day,'Close'] = avg_close_price
                    mcap = data["Close"][-2] * data["Volume"][-2]
                    delta = black_litterman.market_implied_risk_aversion(data['Close'])
                    if (np.max(data.Close)/np.min(data.Close) < 20):
                        self.stock_prices = pd.concat([self.stock_prices, pd.DataFrame({symbol: data.Close})], axis=1)
                        self.details.append([symbol,mcap,delta,np.array(self.holding["Holding"])[3]])
                        print(symbol + " passed")
                    else:
                        print(symbol + " failed")

        if self.save_output:
            self.stock_prices.to_csv(os.path.join(self.output_path,self.name + '_stock_prices.csv'))
            with open(os.path.join(self.output_path,self.name + '_details.csv'), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(self.details)
            
    
    def optimise_portfolio(self, use_filter = True):
        self.stock_prices.fillna(value=self.stock_prices.mean(), inplace = True)
        self.stock_prices = self.stock_prices.dropna(axis = 1, how = 'any')
        
        risk_free_rate = 0.02
        risk_aversion = 0.5
        
        self.filter_stock_prices()
        self.holding = self.holding[self.holding.Symbol.isin(self.fil_stock_prices.keys())]
        if use_filter:
            self.mu = expected_returns.mean_historical_return(self.fil_stock_prices)
            self.S = risk_models.sample_cov(self.fil_stock_prices)
        else:
            self.mu = expected_returns.mean_historical_return(self.stock_prices)
            self.S = risk_models.sample_cov(self.stock_prices)
        
        self.index_return = self.stock_prices.sum(axis=1)[-1]/self.stock_prices.sum(axis=1)[0]
        self.index_price = self.stock_prices.sum(axis=1)
        returns = self.stock_prices.pct_change().dropna()
        # Calculate expected returns and sample covariance
        #calculate old portfolio position
        # Optimise for maximal Sharpe ratio
        ef = EfficientFrontier(self.mu, self.S)
        
        # raw_weights = ef.max_quadratic_utility(risk_aversion)
        raw_weights = ef.max_sharpe()
        
        cleaned_weights = ef.clean_weights()
        ef.save_weights_to_file(os.path.join(self.output_path,"weights.csv"))  # saves to file
        print(cleaned_weights)
        ef.portfolio_performance(verbose=True)
        self.ef = ef
        self.weights = cleaned_weights
        self.rebalance_weight()

    
    def calculate_allocation(self, cash):
        # if you have cash to allocate to a set of stocks, this function will return how to allocate that
        # see rebalance_weight to identify most suitable portfolio
        self.cash = cash
        ef = EfficientFrontier(self.mu, self.S)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        latest_prices = get_latest_prices(self.stock_prices)
        da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=cash)
        allocation, leftover = da.lp_portfolio()
        print("Discrete allocation:", allocation)
        print("Funds remaining: ${:.2f}".format(leftover))
    
    def plot_stock_insights(self, scatter=False):

        ef = EfficientFrontier(self.mu, self.S)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        cla = CLA(self.mu, self.S)
        try:
            eff_front = plotting.plot_efficient_frontier(cla)
            eff_front.tick_params(axis='both', which='major', labelsize=5)
            eff_front.tick_params(axis='both', which='minor', labelsize=5)
            eff_front.get_figure().savefig(os.path.join(self.output_path,"efficient_frontier.png"), dpi = 300)
        except:
            print("Failed to plot efficient frontier")
        cov = plotting.plot_covariance(self.S)
        weights_bar = plotting.plot_weights(cleaned_weights)
        if self.save_output:
            cov.tick_params(axis='both', which='major', labelsize=5)
            cov.tick_params(axis='both', which='minor', labelsize=5)
            cov.get_figure().savefig(os.path.join(self.output_path,"cov_matrix.png"), dpi = 300)
            weights_bar.tick_params(axis='both', which='major', labelsize=5)
            weights_bar.tick_params(axis='both', which='minor', labelsize=5)
            weights_bar.get_figure().savefig(os.path.join(self.output_path,"weights_bar.png"), dpi = 300)
            
        retscomp = self.stock_prices.pct_change()
        corrMatrix = retscomp.corr()
        corr_heat = sns.heatmap(corrMatrix, xticklabels=True, yticklabels=True)
        plt.title("Corr heatmap")
        if self.save_output:
            corr_heat.tick_params(axis='both', which='major', labelsize=5)
            corr_heat.tick_params(axis='both', which='minor', labelsize=5)
            fig = corr_heat.get_figure()
            fig.figsize = (10, 10)
            fig.savefig(os.path.join(self.output_path,"corr_heatmap.png"), dpi = 300)
        plt.show()
        if scatter:
            plt.figure()
            plt.title("Corr scatter")
            self.scattermat = pd.plotting.scatter_matrix(retscomp, diagonal='kde', figsize=(10, 10))
            
            if self.save_output:
                plt.savefig(os.path.join(self.output_path,"corr_scatter.png"), dpi = 300)
            plt.show()
        
    def visualise_holding(self):
        #independent of cash
        holding_position = copy.deepcopy(self.fil_stock_prices)
        new_holding_position = copy.deepcopy(self.fil_stock_prices)
        
        for stock in self.fil_stock_prices.keys():
            share_price = self.fil_stock_prices[stock]
            quantity_held = self.holding.Holding[self.holding.Symbol == stock].values
            new_quantity_held = self.holding.NewHolding[self.holding.Symbol == stock].values
            holding_position[stock] = share_price * quantity_held
            new_holding_position[stock] = share_price * new_quantity_held
            
        dates_dt = self.fil_stock_prices.index
        dates = matplotlib.dates.date2num(dates_dt)
        
        plt.figure(figsize = (15, 5))
        
        plt.plot(dates, holding_position.sum(axis=1)/holding_position.sum(axis=1)[0],label='Old')
        plt.plot(dates, new_holding_position.sum(axis=1)/new_holding_position.sum(axis=1)[0], label='New')
        plt.title("Old vs New positions - start position = %2f"%holding_position.sum(axis=1)[0])
        ax = plt.gca()
        ax.xaxis.set_minor_locator(matplotlib.dates.MonthLocator())
        ax.xaxis.set_minor_formatter(matplotlib.dates.DateFormatter('%m'))
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
        plt.legend()
        
        if self.save_output:
            plt.savefig(os.path.join(self.output_path, "holdings.png"), dpi = 300) 
        plt.show()
    
    def rebalance_weight(self):
        # to calculate the changes needed for an optimum portfolio
        current_stocks = self.holding.Holding
        current_weights = current_stocks/np.sum(current_stocks)

        change_needed_weights = list(self.weights.values()) - current_weights 
        
        change_needed_stocks = change_needed_weights * current_stocks
        self.holding["Change for optimum"] = change_needed_stocks
        self.holding["NewHolding"] = current_stocks + change_needed_stocks
        print("rebalancing...")
        print(change_needed_stocks)
        
    
    def filter_stock_prices(self):
        # soft low pass the higher Wn, the more high freq signals
        b, a = dsp.butter(5, 0.3)
        zi = dsp.lfilter_zi(b, a)
        self.fil_stock_prices = copy.deepcopy(self.stock_prices)
        for i in self.stock_prices:
            t = self.stock_prices.index
            xn = self.stock_prices[i]
            z, _ = dsp.lfilter(b, a, xn, zi=zi*xn[0])
            z2, _ = dsp.lfilter(b, a, z, zi=zi*z[0])
            y = dsp.filtfilt(b, a, xn)
            self.fil_stock_prices[i] = y
            
    # reference for filter code
    def plot_stock_prices(self):
        for i in self.stock_prices:
            t =  matplotlib.dates.date2num(self.stock_prices.index)
            xn = self.stock_prices[i]
            y = self.fil_stock_prices[i]
            
            plt.figure(figsize = (15, 5))
            plt.plot(t, xn, 'b', alpha=0.75)
            plt.plot(t, y, 'k')
            plt.plot(t, xn[0]*self.index_price/self.index_price[0], 'r')
            plt.legend(('noisy signal',
                       'filtfilt',
                       'index',), loc='best')
            plt.grid('major')
            plt.grid('minor')
            plt.title(i)
            ax = plt.gca()
            ax.xaxis.set_minor_locator(matplotlib.dates.MonthLocator())
            ax.xaxis.set_minor_formatter(matplotlib.dates.DateFormatter('%m'))
            ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
            if self.save_output:
               plt.savefig(os.path.join(self.output_path, i + "_price.png"), dpi = 300) 
            plt.show()
            
    def signal_fft(self):
        # TO DO use fft filtered signal
        length = self.stock_prices.shape[0]
        N = 1 * length
        for i in self.stock_prices:
            x = self.stock_prices[i]
            fil_x = self.fil_stock_prices[i]
            y = fft(list(x))
            fil_y = fft(list(fil_x))
            xf = fftfreq(length, 1 / 1)
            self.xf=xf
            # ignore high frequency, cut off at 10 day periodicity 1/0.1
            plt.figure(figsize = (15, 5))

            plt.bar(xf[np.logical_and(xf>0.0001, xf<0.1)], np.abs(fil_y[np.logical_and(xf>0.0001, xf<0.1)]),width=0.0005)
            plt.legend('signal spectrum', loc='best')

            plt.title("FFT - " + i)
            plt.xlabel("1/days")
            plt.grid(True)
            if self.save_output:
               plt.savefig(os.path.join(self.output_path, i + "_fft.png"), dpi = 300) 
            plt.show()
            
    def which_stock(self):
        returns = self.stock_prices.pct_change()
        mean_daily_returns = returns.mean()
        volatilities = returns.std()
        # plt.figure(figsize = (10, 10))
        self.combine = pd.DataFrame({'returns': mean_daily_returns * self.stock_prices.shape[0],
                       'volatility': volatilities * self.stock_prices.shape[0]})
        g = sns.jointplot("volatility", "returns", data=self.combine, kind="reg",height=7)
        for i in range(self.combine.shape[0]):
            plt.annotate(self.combine.index[i], (self.combine.iloc[i, 1], self.combine.iloc[i, 0]),fontsize=5)
        plt.text(0, -1.5, 'SELL', fontsize=10)
        plt.text(0, 1.0, 'BUY', fontsize=10)
        plt.yticks(fontsize=5, rotation=0)
        plt.xticks(fontsize=5, rotation=45)
        if self.save_output:
            plt.savefig(os.path.join(self.output_path, "which_stock.png"), dpi=300)
        plt.show()
                            
    def calculate_accuracy(self, real, predict):
        real = np.array(real) + 1
        predict = np.array(predict) + 1
        percentage = 1 - np.sqrt(np.mean(np.square((real - predict[0:len(real)]) / real)))
        return percentage * 100
    
    def anchor(self, signal, weight):
        buffer = []
        last = signal[0]
        for i in signal:
            smoothed_val = last * weight + (1 - weight) * i
            buffer.append(smoothed_val)
            last = smoothed_val
        return buffer
    
    # load right dates
    def get_forecast_dates(self, results):
        d = self.start
        end = d + dt.timedelta(days = 2000)
        step = dt.timedelta(days=1)
        dates_dt = []
        while d < end:
            #if not weekend
            if not d.isoweekday() in [6,7]:
                dates_dt.append(d.strftime('%Y-%m-%d'))
            d += step
        dates_dt = dates_dt[0:len(results[0])]
        return dates_dt
    
        
    def monte_carlo_drift(self, forecast_length):
        number_simulation = 100
        forecast_length = 50
        for stock in self.stock_prices:
            
            close = self.fil_stock_prices[stock]
            returns = pd.DataFrame(close).pct_change()
            last_price = close[-1]
            results = pd.DataFrame()
            avg_daily_ret = returns.mean()
            variance = returns.var()
            daily_vol = returns.std()
            daily_drift = avg_daily_ret - (variance / 2)
            drift = daily_drift - 0.5 * daily_vol ** 2
            
            results = pd.DataFrame()
            
            for i in tqdm(range(number_simulation)):
                prices = []
                prices.append(close[-1])
                for d in range(forecast_length):
                    shock = [drift + daily_vol * np.random.normal()]
                    shock = np.mean(shock)
                    price = prices[-1] * np.exp(shock)
                    prices.append(price)
                results[i] = prices
            raveled = results.values.ravel()
            raveled.sort()
            cp_raveled = raveled.copy()
            
            plt.figure(figsize=(17,5))
            plt.subplot(1,3,1)
            plt.plot(results)
            plt.ylabel('Value')
            plt.xlabel('Simulated days')
            plt.subplot(1,3,2)
            sns.distplot(close,norm_hist=True)
            plt.title(stock + ': $\mu$ = %.2f, $\sigma$ = %.2f'%(close.mean(),close.std()))
            plt.subplot(1,3,3)
            sns.distplot(raveled,norm_hist=True,label='monte carlo samples')
            sns.distplot(close,norm_hist=True,label='real samples')
            plt.title(stock + ': simulation $\mu$ = %.2f, $\sigma$ = %.2f'%(raveled.mean(),raveled.std()))
            plt.legend()
            if self.save_output:
                plt.savefig(os.path.join(self.output_path, stock + "_mc_drift.png"), dpi=300)
            plt.show()
            
            
            
    def lstm_forecast(self, epoch, dropout_rate, learning_rate, simulation_size, test_size):
        df = self.fil_stock_prices
        sns.set()
        tf.compat.v1.random.set_random_seed(111)
        
        # test_size = 60
        # simulation_size = 10
        num_layers = 1
        size_layer = 128
        timestamp = 5
        # dropout_rate = 0.8
        future_day = test_size
        # forecast_day
        # learning_rate = 0.01
        # epoch = 100
        
        for s in range(df.shape[1]):
            key = df.keys()[s]
            print(key)
            print("forecast progress : " + str(s*100/df.shape[1]) + " %") 
            minmax = MinMaxScaler().fit(df.iloc[:, s:s+1].astype('float32')) # Close index
            df_log = minmax.transform(df.iloc[:, s:s+1].astype('float32')) # Close index
            df_log = pd.DataFrame(df_log)
            df_log.head()
            df_log = df_log.dropna()

            ## for testing accuracy of this model
            # df_train = df_log.iloc[:-test_size]
            # df_test = df_log.iloc[-test_size:]
            # df.shape, df_train.shape, df_test.shape

            ## syn data generation
            df_train = df_log
            
            tf.reset_default_graph()
            modelnn = lstm.lstm(learning_rate, num_layers, df_log.shape[1], size_layer, df_log.shape[1], dropout_rate)
            
            results = []
            for i in range(simulation_size):
                print('simulation %d'%(i + 1))
                results.append(lstm.lstm.forecast(
                        self,
                        modelnn, 
                        epoch,
                        num_layers,
                        size_layer,
                        df,
                        df_train,
                        timestamp,
                        test_size,
                        minmax,
                        ))
            date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()
            self.results = results
            
            for i in range(test_size):
                date_ori.append(date_ori[-1] + timedelta(days = 1))
            date_ori = pd.Series(date_ori).dt.strftime(date_format = '%Y-%m-%d').tolist()
            date_ori[-5:]
            
            accepted_results = []
            for r in results:
                # if (np.array(r[-test_size:]) < np.min(df[key].iloc[-test_size:].values)).sum() / 2 == 0 and \
                # (np.array(r[-test_size:]) > np.max(df[key].iloc[-test_size:].values) * 2).sum() == 0:
                if (np.array(r[-test_size:]) < 0).sum() == 0 and \
                (np.array(r[-test_size:]) > np.max(df[key].iloc[-test_size:].values) * 6).sum() == 0:
                    accepted_results.append(r)
                    
            self.accepted_results = accepted_results
            
            accuracies = [self.calculate_accuracy(df[key].iloc[-test_size:].values, r) for r in results]
            
            plt.figure(figsize = (15, 5))
            
            
            std_dev = np.array(self.accepted_results).std(axis=0)
            average = np.array(self.accepted_results).mean(axis=0)
            ci = 1.96 * std_dev
            dates_dt = self.get_forecast_dates(results)
            
            # find best matched forecast
            errors = []            
            for res in self.accepted_results:
                error = sum((res - average) ** 2)
                errors.append(error)
                
            # print([errors == min(errors)][0])
            # best_forecast = self.accepted_results[np.array([errors == min(errors)]).astype(int)]
            
            
            
            self.lstm_forecast_price['Dates'] = pd.to_datetime(dates_dt)
            self.lstm_forecast_price.set_index('Dates')
            
            ax = plt.gca()
            dates = matplotlib.dates.date2num(dates_dt)
            ax.xaxis.set_minor_locator(matplotlib.dates.MonthLocator())
            ax.xaxis.set_minor_formatter(matplotlib.dates.DateFormatter('%m'))
            ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
            # self.forecast_stock_prices[key.split('.')[0]] = self.accepted_results
            
            
            
            try:
                # try to save best forecast in parent class
                best_forecast = np.array(accepted_results)[errors == min(errors)][0]
                self.lstm_forecast_price[key] = best_forecast

                # try to display as aggregated
                plt.plot(dates,best_forecast, label="best forecast", color = 'r')
                plt.plot(dates,average,label="avg forecast", color = 'b')
                plt.fill_between(dates, (average-ci), (average+ci), label = '95% confidence', color='b', alpha=.5)
            except:
                # plot acceptable ones
                for no, r in enumerate(accepted_results):
                    plt.plot(dates, r, label = 'forecast %d'%(no + 1))
                    
            
                    
                    
            # plt.plot(dates[:len(df[key])-test_size], df[key][:-test_size], label = 'true trend (train)', c = 'green')
            # plt.plot(dates[len(df[key])-test_size-1:len(df[key])], df[key][-test_size-1:], label = 'true trend (test)', c = 'red')
            
            # forecasting
            plt.plot(dates[:len(df[key])], df[key], label = 'true trend (train)', c = 'k')
            plt.legend()
            
            plt.title(key.split('.')[0] + ' average accuracy: %.2f'%(np.mean(accuracies)))
            x_range_future = np.arange(len(results[0]))
            # plt.xticks(x_range_future[::30], date_ori[::30])
            if self.save_output:
                plt.savefig(os.path.join(self.output_path, "forecast_LSTM_" + key.split('.')[0]+'.png'), dpi=300)
            plt.show()
        if self.save_output:
           self.stock_prices.to_csv(os.path.join(self.output_path,self.name + '_lstm_stock_prices.csv'))
        
            
    def gru2_forecast(self, epoch, dropout_rate, learning_rate, simulation_size, test_size):
        df = self.fil_stock_prices
        sns.set()
        tf.compat.v1.random.set_random_seed(1234)
        
        # test_size = 60
        # simulation_size = 10
        num_layers = 1
        size_layer = 128
        timestamp = 5
        # dropout_rate = 0.8
        future_day = test_size
        # forecast_day
        # learning_rate = 0.01
        # epoch = 100
        
        for s in range(df.shape[1]):
            key = df.keys()[s]
            print(key)
            print("forecast progress : " + str(s*100/df.shape[1]) + " %")                        

            minmax = MinMaxScaler().fit(df.iloc[:, s:s+1].astype('float32')) # Close index
            df_log = minmax.transform(df.iloc[:, s:s+1].astype('float32')) # Close index
            df_log = pd.DataFrame(df_log)
            df_log.head()
            df_log = df_log.dropna()

            ## for testing accuracy of this model
            # df_train = df_log.iloc[:-test_size]
            # df_test = df_log.iloc[-test_size:]
            # df.shape, df_train.shape, df_test.shape

            ## syn data generation
            df_train = df_log

            tf.reset_default_graph()
            modelnn = gru2.gru2(learning_rate, num_layers, df_log.shape[1], size_layer, df_log.shape[1], dropout_rate)
            
            results = []
            for i in range(simulation_size):
                print('simulation %d'%(i + 1))
                results.append(gru2.gru2.forecast(
                        self,
                        modelnn, 
                        epoch,
                        num_layers,
                        size_layer,
                        df,
                        df_train,
                        timestamp,
                        test_size,
                        minmax,
                        ))
            date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()
            self.results = results
            
            for i in range(test_size):
                date_ori.append(date_ori[-1] + timedelta(days = 1))
            date_ori = pd.Series(date_ori).dt.strftime(date_format = '%Y-%m-%d').tolist()
            date_ori[-5:]
            
            accepted_results = []
            for r in results:
                # if (np.array(r[-test_size:]) < np.min(df[key].iloc[-test_size:].values)).sum() / 2 == 0 and \
                # (np.array(r[-test_size:]) > np.max(df[key].iloc[-test_size:].values) * 2).sum() == 0:
                if (np.array(r[-test_size:]) < 0).sum() == 0 and \
                (np.array(r[-test_size:]) > np.max(df[key].iloc[-test_size:].values) * 6).sum() == 0:
                    accepted_results.append(r)
                    
            self.accepted_results = accepted_results
            
            accuracies = [self.calculate_accuracy(df[key].iloc[-test_size:].values, r) for r in results]
            
            plt.figure(figsize = (15, 5))
            
            std_dev = np.array(self.accepted_results).std(axis=0)
            average = np.array(self.accepted_results).mean(axis=0)
            ci = 1.96 * std_dev
            dates_dt = self.get_forecast_dates(results)
            
            # find best matched forecast
            errors = []            
            for res in self.accepted_results:
                error = sum((res - average) ** 2)
                errors.append(error)
                
            # print([errors == min(errors)][0])
            # best_forecast = self.accepted_results[np.array([errors == min(errors)]).astype(int)]
            
            
            
            self.lstm_forecast_price['Dates'] = pd.to_datetime(dates_dt)
            self.lstm_forecast_price.set_index('Dates')
            
            ax = plt.gca()
            dates = matplotlib.dates.date2num(dates_dt)
            ax.xaxis.set_minor_locator(matplotlib.dates.MonthLocator())
            ax.xaxis.set_minor_formatter(matplotlib.dates.DateFormatter('%m'))
            ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
            # self.forecast_stock_prices[key.split('.')[0]] = self.accepted_results

            
            try:
                # try to save best forecast in parent class
                best_forecast = np.array(accepted_results)[errors == min(errors)][0]
                self.gru2_forecast_price[key] = best_forecast
                
                # try to display as aggregated
                plt.plot(dates,average,label="forecast")
                plt.fill_between(dates, (average-ci), (average+ci), label = '95% confidence', color='b', alpha=.5)
            except:
                # plot acceptable ones
                for no, r in enumerate(accepted_results):
                    plt.plot(dates, r, label = 'forecast %d'%(no + 1))
            # plt.plot(dates[:len(df[key])-test_size], df[key][:-test_size], label = 'true trend (train)', c = 'green')
            # plt.plot(dates[len(df[key])-test_size-1:len(df[key])], df[key][-test_size-1:], label = 'true trend (test)', c = 'red')
            
            # forecasting
            plt.plot(dates[:len(df[key])], df[key], label = 'true trend (train)', c = 'green')
            
            plt.legend()
            
            plt.title(key.split('.')[0] + ' average accuracy: %.2f'%(np.mean(accuracies)))
            x_range_future = np.arange(len(results[0]))
            # plt.xticks(x_range_future[::30], date_ori[::30])
            
            if self.save_output:
                plt.savefig(os.path.join(self.output_path, "forecast_GRU2_" + key.split('.')[0]+'.png'), dpi=300)
            plt.show()
        if self.save_output:
            self.stock_prices.to_csv(os.path.join(self.output_path,self.name + '_gru2_stock_prices.csv'))
    
    def dcnn_forecast(self, epoch, dropout_rate, learning_rate, simulation_size, test_size):
        df = self.fil_stock_prices 
        sns.set()
        tf.compat.v1.random.set_random_seed(1234)
        
        # test_size = 60
        # simulation_size = 10
        num_layers = 1
        size_layer = 128
        timestamp = 5
        # dropout_rate = 0.8
        future_day = test_size
        # forecast_day
        # learning_rate = 0.01
        # epoch = 100
        
        for s in range(df.shape[1]):
            key = df.keys()[s]
            print(key)
            print("forecast progress : " + str(s*100/df.shape[1]) + " %") 
            minmax = MinMaxScaler().fit(df.iloc[:, s:s+1].astype('float32')) # Close index
            df_log = minmax.transform(df.iloc[:, s:s+1].astype('float32')) # Close index
            df_log = pd.DataFrame(df_log)
            df_log.head()
            df_log = df_log.dropna()

            ## for testing accuracy of this model
            # df_train = df_log.iloc[:-test_size]
            # df_test = df_log.iloc[-test_size:]
            # df.shape, df_train.shape, df_test.shape

            ## syn data generation
            df_train = df_log

            tf.reset_default_graph()
            modelnn = dcnn.dcnn(
                learning_rate, num_layers, df_log.shape[1], size_layer, df_log.shape[1], 
                dropout = dropout_rate
                    )
            
            results = []
            for i in range(simulation_size):
                print('simulation %d'%(i + 1))
                results.append(dcnn.dcnn.forecast(
                        self,
                        modelnn, 
                        epoch,
                        num_layers,
                        size_layer,
                        df,
                        df_train,
                        timestamp,
                        test_size,
                        minmax,
                        ))
            date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()
            self.results = results
            
            for i in range(test_size):
                date_ori.append(date_ori[-1] + timedelta(days = 1))
            date_ori = pd.Series(date_ori).dt.strftime(date_format = '%Y-%m-%d').tolist()
            date_ori[-5:]
            
            accepted_results = []
            for r in results:
                # if (np.array(r[-test_size:]) < np.min(df[key].iloc[-test_size:].values)).sum() / 2 == 0 and \
                # (np.array(r[-test_size:]) > np.max(df[key].iloc[-test_size:].values) * 2).sum() == 0:
                if (np.array(r[-test_size:]) < 0).sum() == 0 and \
                (np.array(r[-test_size:]) > np.max(df[key].iloc[-test_size:].values) * 4).sum() == 0:
                    accepted_results.append(r)
                    
            self.accepted_results = accepted_results
            
            accuracies = [self.calculate_accuracy(df[key].iloc[-test_size:].values, r) for r in results]
            
            plt.figure(figsize = (15, 5))
            
            std_dev = np.array(self.accepted_results).std(axis=0)
            average = np.array(self.accepted_results).mean(axis=0)
            ci = 1.96 * std_dev
            dates_dt = self.get_forecast_dates(results)
            
            # find best matched forecast
            errors = []            
            for res in self.accepted_results:
                error = sum((res - average) ** 2)
                errors.append(error)
                
            # print([errors == min(errors)][0])
            # best_forecast = self.accepted_results[np.array([errors == min(errors)]).astype(int)]
            
            
            
            self.lstm_forecast_price['Dates'] = pd.to_datetime(dates_dt)
            self.lstm_forecast_price.set_index('Dates')
            
            ax = plt.gca()
            dates = matplotlib.dates.date2num(dates_dt)
            ax.xaxis.set_minor_locator(matplotlib.dates.MonthLocator())
            ax.xaxis.set_minor_formatter(matplotlib.dates.DateFormatter('%m'))
            ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
            # self.forecast_stock_prices[key.split('.')[0]] = self.accepted_results
            
            try:
                # try to save best forecast in parent class
                best_forecast = np.array(accepted_results)[errors == min(errors)][0]
                self.dcnn_forecast_price[key] = best_forecast
                
                # try to display as aggregated
                plt.plot(dates,average,label="forecast")
                plt.fill_between(dates, (average-ci), (average+ci), label = '95% confidence', color='b', alpha=.5)
            except:
                # plot acceptable ones
                for no, r in enumerate(accepted_results):
                    plt.plot(dates, r, label = 'forecast %d'%(no + 1))
                    
            # plt.plot(dates[:len(df[key])-test_size], df[key][:-test_size], label = 'true trend (train)', c = 'green')
            # plt.plot(dates[len(df[key])-test_size-1:len(df[key])], df[key][-test_size-1:], label = 'true trend (test)', c = 'red')
            
            # forecasting
            plt.plot(dates[:len(df[key])], df[key], label = 'true trend (train)', c = 'green')
            
            plt.legend()
            
            plt.title(key.split('.')[0] + ' average accuracy: %.2f'%(np.mean(accuracies)))
            x_range_future = np.arange(len(results[0]))
            # plt.xticks(x_range_future[::30], date_ori[::30])
            if self.save_output:
                plt.savefig(os.path.join(self.output_path, "forecast_DCNN_" + key.split('.')[0]+'.png'), dpi=300)
            plt.show()
        if self.save_output:
            self.stock_prices.to_csv(os.path.join(self.output_path,self.name + '_dcnn_stock_prices.csv'))