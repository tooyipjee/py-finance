from runAnalysis import run_analysis 

# point to directory that main.py is being run from
wd = "C:/Users/JYT1/Documents/GitHub/personal-finance/"

# how many days into the future to perform analysis
forecast_length = 50

# how much cash is at hand, for discrete allocation calculation
cash = 100000

# how many years of data to accumulate for training and viz
years_of_data = 1

# MY Symbols
country = "Malaysia"
symbols_files = ['trust_stocks', 'bursa_ftse_100_stocks']
# symbols_files = ['test']

for symbols in symbols_files:
    run_analysis(symbols, country, wd, forecast_length, cash, years_of_data)


# UK Symbols
country = "United Kingdom"
symbols_files = ['ftse_100','ftse_250','penny_stocks']

for symbols in symbols_files:
    run_analysis(symbols, country, wd, forecast_length, cash, years_of_data)