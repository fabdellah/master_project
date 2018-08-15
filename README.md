# Mater Project: Index Prediction and Asset Optimization: The Case of Russian Gas Supplies to Europe

In this project, we want first to predict the Russian gas index with finding the unknown parameters used in the Russian pricing formula. Then, in the second part of the project, we want to use these predictions of the Russian gas price to optimize Europe gas imports from Russia. A Monte Carlo model will be used for this part. 

# Author

Farah Abdellah

    
# Project Specification

All code is found in the project directory: Master Project

    - code: contains all python files and the run.py
    - data: contains all data requested to run the code
    - ReadMe: if you are reading that... you got it :)
    
# Dependencies

The following dependencies will need to be installed in order to run the project:

#-> Libraries

* [Anaconda3] - Download and install Anaconda with python3
* [Scikit-Learn] - Download scikit-learn library with conda

    ``pip install scikit-learn``

* [SciPy] - Install SciPy library 

    ``pip install scipy``
    

#-> Data Set

@ Exteral_Data.xls

    Contains the historical Russian gas prices.

@ spot_prices.xls

    Contains the historical spot prices of Oil, Power, Coal and market gas (TTF).


# Running the code and Reproducing results 

- Choose Spyder on Anaconda navigator

- Make sure to set the working directory to ".../Master Project"

- Keep the parameters as they are and run the code by steps:

      - Step 0: - Open and run the files load_data.py, class_gas_storage.py, class_alternate.py, 
                  functions_model_calibration.py, functions_gas_storage.py and functions_predictions.py
                  
                - Open the file run.py and run it by steps as explained below
      
      - Step 1: Load the data
      
      - Step 2: Predict the Russian gas price
      
                  - Train the model on the training set of data and get the optimal parameters (coefficients, lags,            
                    ma_periods and reset periods)
                   
                  - Test the model on the testing set of data and plot the predictions. The predictions (i.e forward curve of 
                    Russia gas price) are saved in an Excel file "y_pred_test.xlsx"
                    
                    Remark: The training part takes approximatively 5 hours to get accurate results. The number of iterations 
                    should be > 45. In order to avoid long processing time, the accurate parameters have been used and saved 
                    in the file "y_pred_test.xlsx". This accurate result is used later in the code. 
                  
      - Step 3: Model Calibration
      
                  - Calibrate the 1-factor Schwartz model to the actual forward curve 
                  
                  - Make simulations. The simulated paths for prices are saved in an excel file 
                    "simulated_price_matrix_spot.xlsx"
      
      
      - Step 4: Monte Carlo for gas storage
      
                  - Specify the parameters for the gas storage class: The parameters in the code are the ones used to get the     
                    final results of the report
                    
                    Remark: You can change the parameter 'simulations' to 'example' if you want to get the general gas storage 
                    examples shown in the report (Cantongo case, Section 6.2.1 in the report)
                    
                  - Compute the optimal volume to import and plot the results 
                   
          

# OS Requirements

    - Tested on MacOSX and Window10
    - 16 GO RAM is recommended to reproduce our best solution 
    

# Time

    Approximated run time on a 8 cores, 3GHZ, 32 GO RAM computer 
    
          Step                                         Time
    - Load the data:                               few seconds
    - Predict Russia's gas price:           5 hours (for 45 iterations)
    - Model calibration:                           few seconds
    - Monte Carlo for gas storage:         < 4 minutes (for 10000 simulations)

@ You can also change the parameters if you want to try other combinations.
