""" Stock Price Predictor Using Machine Learning In Python"""
        """Inro to Problem Solving in Python 2024"""
            """Created by: Beckett Karow"""







import yfinance as yf     #imports stock data from yahoo finance
import matplotlib.pyplot as plt     #imports matplot lib to visualize stock prediction data
from sklearn import linear_model
model = linear_model.LinearRegression()         #Importing linear regression model which will be used to predict prices
from sklearn.model_selection import train_test_split      #importing this allows us to split the data into training data and testing data
import pandas                 #importing pandas in order to manipulate data from yahoo finance

def PredictPrice(ticker):         #creating function to predict stock price
    PriceData = yf.download(ticker, period = "1mo")     #downloads stock data (open, close, high, low, etc...)
    x = PriceData[["Open", "High", "Low"]]          #data that will be used in the model
    y = PriceData["Close"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)   #Splitting the data up into training data and testing data


    model.fit(x_train, y_train)          #finding the line of best fit for the training data
    

    yesterday = PriceData.iloc[-1]         #Accessing data from the previous day
    prediction = model.predict([[yesterday["Open"], yesterday["High"], yesterday["Low"]]])        #predicting the price for tomorrow based on the latest data
    print("The predicted price for", ticker, "tomorrow is:", prediction[0])



    plt.plot(PriceData.index, PriceData["Close"], label = "Actual Closing Price")
    plt.axhline(prediction, color = "black", linestyle = "dashed", label = "Predicted Price For Tomorrow: " + str(prediction[0]))
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(ticker + " Price Prediction For Tomorrow")               #Creating plot
    plt.legend()
    plt.show()
    

#PredictPrice("MSFT")
#PredictPrice("AMZN")
#PredictPrice("COST")
    
    

    


    
