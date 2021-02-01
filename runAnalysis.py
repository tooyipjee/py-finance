import os
from Portfolio import Portfolio

# symbols = 'bursa_ftse_100_stocks'
# symbols = 'ftse_100'

# symbols = 'trust_stocks'
def run_analysis(symbols, country, wd, forecast_length, cash, years_of_data):
    
    use_filter = True
    
    cwd = os.getcwd()
    # define the name of the directory to be created
    path = os.path.join(cwd, "outputs\\" + symbols)
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
    
    # code for creation
    ori_portfolio = Portfolio()
    ori_portfolio.create_portfolio(years_of_data, wd + "symbols/"+ symbols +".csv", path)
    ori_portfolio.download_stocks("investing", country)
    ori_portfolio.save_output = True
    ori_portfolio.optimise_portfolio(use_filter = use_filter)
    
    # Portfolio creation
    ori_portfolio.calculate_allocation(cash) # some times in pence/cents/sens
    ori_portfolio.visualise_holding()
    ori_portfolio.which_stock()
    
    # Risk analysis
    ori_portfolio.monte_carlo_drift(forecast_length)  
    
    # Insights
    ori_portfolio.plot_stock_insights()
    ori_portfolio.plot_stock_prices()
    ori_portfolio.signal_fft()
    
    ori_portfolio.lstm_forecast(
        epoch=150, 
        dropout_rate=0.75, 
        learning_rate=0.01, 
        simulation_size = 20,
        test_size = forecast_length,
        )
    
    # ori_portfolio.dcnn_forecast(
    #     epoch=300, 
    #     dropout_rate=0.75, 
    #     learning_rate=5e-4, 
    #     simulation_size = 10,
    #     test_size = forecast_length,
    #     )
    
    
    # ori_portfolio.gru2_forecast(
    #     epoch=150, 
    #     dropout_rate=0.75, 
    #     learning_rate=0.01, 
    #     simulation_size = 20,
    #     test_size = forecast_length,
    #     )
    
    
    ### synthetic portfolio
    # code for creation
    path = os.path.join(cwd, "outputs\\" + symbols + "\\synthetic")
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
        
    syn_portfolio = Portfolio()
    syn_portfolio.create_portfolio(years_of_data,  wd + "symbols/"+ symbols +".csv", path)
    syn_portfolio.stock_prices = ori_portfolio.lstm_forecast_price.set_index("Dates").dropna(axis=1)
    syn_portfolio.save_output = True
    syn_portfolio.optimise_portfolio(use_filter = use_filter)
    
    syn_portfolio.visualise_holding()
    syn_portfolio.which_stock()
    syn_portfolio.calculate_allocation(cash)
    
    ori_portfolio.plot_stock_insights()
    syn_portfolio.plot_stock_prices()
    syn_portfolio.signal_fft()