# Bike Sharing Demand Prediction Supervised ML Regression
## Abstract 
Bike sharing has significantly increased urban mobility and is a practical, affordable, and environmentally friendly mode of transportation. However, due to the asymmetric user demand distribution and the scarcity of shared bikes, docks, or parking spaces, it is frequently very challenging to achieve a balanced utilization of shared bikes. It will be easier for operating agencies to rebalance bike-sharing systems quickly and effectively if we can forecast the short-term demand for bike-sharing.

## INTRODUCTION
Currently, Rental bikes are introduced in many urban cities for the enhancement of mobility comfort. It is important to make the rental bike available and accessible to the public at the right time as it lessens the waiting time. Eventually, providing the city with a stable supply of rental bikes becomes a major concern. The crucial part is the prediction of the bike count required at each hour for the stable supply of rental bikes.

## Problem Statement
We are here to explore the bike-sharing dataset, and by using the historical bike-sharing data, we have to build a machine-learning model to predict the demand for bikes each hour. 
The demand forecast can solve the following major problems.
1.	It will help to maximize the availability of bikes to the customer and minimize waiting time to get a bike.
2.	It will minimize the cost to the business that is required to increase availability and also will provide scalability to the businesses.

## PROJECT ARCHITECTURE - STEPS INVOLVED
This Project is segregated broadly into the below sections.
1.	Data Understanding and Cleaning 
2.	EDA
3.	Feature Engineering 
4.	Model implementation 
5.	Conclusion

## DATA UNDERSTANDING AND CLEANING

### DATA DESCRIPTION
The dataset contains weather information (Temperature, Humidity, Windspeed, Visibility, Dewpoint, Solar radiation, Snowfall, Rainfall), the number of bikes rented per hour and date information
•	Date: year-month-day

•	Rented Bike count: Count of bikes rented at each hour

•	Hour: Hour of the day

•	Temperature: in Celsius

•	Humidity: %

•	Wind Speed: m/s

•	Visibility: 10m

•	Dew point Temperature: Celsius

•	Solar_radiation : MJ/m2

•	Rainfall: mm

•	Seasons: Winter, Spring, Summer, Autumn ∙ Holiday - Holiday/No holiday

•	Functional Day: NoFunc(Non Functional Hours), Fun(Functional hours)

### LOADING LIBRARIES AND DATASET
First, all the necessary libraries like NumPy, Pandas, Seaborn, Matplotlib are loaded that are required to understand and explore the data. The data set then is loaded using read_csv function of the Pandas library.

### DATA UNDERSTANDING
The data is explored using pandas’ functions like head(), tail(), info(), describe(), nunique(), value_counts(), isnull(), duplicated(), etc. The following conclusions are made.
•	The data is properly formatted.

•	There are no null values.

•	There are no duplicate values.

•	There are 3 categorical columns and 10 numerical columns.

### DATA CLEANING
Since the data is already in the proper format, and there were no null values, the data is already clean and thus no operation is required in this step. 


##	EDA
###	UNIVARIABLE ANALYSIS:
•	The Rented Bike count is right-skewed, which means there are only a few times when the demand for bikes is very high.

•	The hours are evenly distributed.

•	The Temperature and Humidity are almost symmetrically distributed.

•	The wind speed is distributed right, which means that there are sometimes when the windspeed is high.

•	The visibility is left-skewed, which means there are sometimes when the visibility is low.

•	The dew point temperature is slightly left-distributed.

•	Solar radiation is right distributed, so there are sometimes when the solar radiation is very high.

•	The Rainfall is also right distributed, so there are sometimes when the rainfall is very high.

•	The snowfall is also right distributed, so there are sometimes when the snowfall is very high. Here most of the variables are skewed, which probably is the reason 
for the skewness in the rented bikes.

•	There are 95.07% times no holiday.

•	The functioning day are 96.63%.

•	There are a total of 4 seasons, each carries the same weightage.
###	BIVARIATE VARIABLE ANALYSIS: 
####	Bar plot of mean rented bikes and season:
•	The demand is very high in summer and lowest in winter.

•	Demand: Summer > Autumn > Spring > Winter

•	The demand in summer is 36.8% and demand in spring is 26% and in Autumn is 29% but demand in winter is just 8.03%.
####	Bar plot of mean demand and Holiday:
•	The average per-hour demand on non-holiday is 58.87% and on Holiday is 41.13%.
####	Bar plot of mean rented bikes and month:
•	The demand is very high in the 6th month and very low in the 1st month.

•	The demand increases from low to high and decreases gradually from high to low.
####	Bar plot of mean rented bikes and hour:
•	The demand is very high around 8 and 18 and the demand is very low around 4, and 10.

•	The demand increases and decreases gradually.

•	The demand is low at 4 and then it gradually increases till 8 and afterwards it gradually decreases till 10 and then again gradually increases till 18 and then gradually decreases till 4.

•	For all the seasons, the demand follows the above pattern. So, it means that the demand for bikes is not much influenced by the change in daily weather
###	MULTIVARIATE ANALYSIS 
#### Weather analysis - Scatter plot: 
•	By scatter plot it is clear that some of the weather parameters are very linear in nature to demand but some are not. To get a clear picture, we can use a correlation heatmap.
#### Weather analysis - Correlation matrix
•	The temperature and Dew point temperature are very highly positively correlated to the rented bikes so the demand for bikes increases as the temperature or dew point temperature increases.

•	Solar radiation and visibility positively correlated to rented bikes, so the demand of bikes increases if solar radiation or visibility increases.

•	The wind speed is little positively correlated to the rented bikes.

•	The rainfall, Snowfall, and Humidity are negatively correlated to the rented bikes, so the demand for bikes decreases if the rainfall, Snowfall, or Humidity increases.

•	Positive correlated: Temperature > Dew point temperature > Solar Radiation > Visibility > Wind

•	Negative correlated: Rainfall > Snowfall > Humidity

##	FEATURE ENGINEERING 

###	HANDLING MULTICOLLINEARITY 
Multicollinearity is spotted using the variance inflation factor. After applying the VIF function for each column, it is found that the data is multicollinear with a VIF score of Temperature = 35 and of Dew point temperature = 17. The multicollinearity in data is removed by dropping the Dew point temperature column from the dataset.

###	FEATURES CREATION
From the EDA it was clear that the demand for bikes depends upon the hours, and season, and also demand changes on weekends therefore three more following features are created.
-	Months: This feature is created to feed the seasonality information in the ML model.
-	Hours: This feature is created to feed the daily pattern information in the ML model.
-	Weekend: This feature is created to feed the weekend information in the ML model.


### OUTLIERS TREATMENT USING FEATURE TRANSFORMATIONS
The yeo-johnson method using the Power Transformer function of Sklearn is used to minimize the skewness in the data and as a byproduct, the outlier issue is also resolved with this transformation.
Also, the data is normalized using standardized (Z-score normalization).
### FEATURE ENCODING   
•	The ordinal encoding is used for holidays and functioning days with the following mapping.
mapping = [{'col': 'Holiday', 'mapping': {'Holiday':1, 'No Holiday':0}}, {'col': 'Functioning Day', 'mapping': {'Yes':1, 'No': 0} }]

•	Since the hour is a categorical column, one-hot encoding is used to encode the hour column. 

•	To avoid excess columns, Binary encoding is used to encode the month column.

###	FEATURE SELECTION

Feature selection algorithms like RFECV and SequentialFeatureSelector are used in the last step. But both methods decreased the test accuracy. Since none of the methods were able to beat the benchmark score therefore, we have not considered it.


##	MODEL IMPLEMENTATION 
###	PRE-PROCESSING DATA
The data is split into two parts by a ratio of 80:20. A total of 80% of the data (7008 rows) is used for training and the rest 20% is used for model validation (testing).
###	MODELS USED
The following models from simple to complex are used to check the behaviour of each model. However, there are many powerful bagging and boosting models that are also used.
•	LinearRegression

•	DecisionTreeRegressor

•	RandomForestRegressor

•	GradientBoostingRegressor

•	XGBRegressor

•	AdaBoostRegressor

•	KNeighborsRegressor

•	HistGradientBoostingRegressor

###	MODEL COMPARISON
Here the R2 metric is used to compare the models, the models are compared on the basis of cross validation R2, test R2, train R2. It is clear from below, that the XGBRegressor and HistGradientBoostingRegressor are the two top models as the test and Cross validation score is high for both models.

###	HYPERPARAMETER TUNING 
Grid Search CV and FLAML are used for hyperparameter tunning of the top two models obtained in the model comparison step. 
•	HistGradientBoostingRegressor(l2_regularization=2.0e-09,
                              learning_rate=0.06, max_iter=512,
                              max_leaf_nodes=18, min_samples_leaf=49,
                              random_state=1,
                              validation_fraction=0.086,
                              warm_start=True)


•	XGBRegressor(eta=0.05, 
             max_depth=10, 
             n_estimators=150,                                 
             objective='reg:squarederror', 
             random_state=3)

###	MODEL EXPLANATION  
• In the case of XGBRegressor, Functioning day, Hour 22, Hour 19, Hour 15, and Rainfall are the top 5 influencing features.

• In the case of random forests, Temperature, Functioning Day, and Humidity, Rainfall, Hour 22 are the top 5 influencing features.


## Conclusions:
1.	HistGradientBoostingRegressor gives the best results as compared to all other models. The backward feature selection does not increase the cross-validation score, but it decreases the test and train accuracy to a great extent, therefore we can avoid it. 
2.	In the case of HistGradientBoostingRegressor, we can’t see the most influencing feature as it does not support the feature importance attribute.
3.	For HistGradientBoostingRegressor, the cross-validation R2 is 0.929, Test R2 is 0.93, and train R2 is 0.96.
4.	XGBRegressor gives the second-best result, with cross-validation R2 of 0.925, Test R2 of 0.93, and train R2 of 0.99. The random forest is the third-best model with a cross-validation R2 score of 0.88, test R2 of 0.89, and train R2 of 0.93.
5.	In the case of XGBRegressor, Functioning day, Hour 22, Hour 19, Hour 15, and Rainfall are the top 5 influencing features.
6.	In the case of random forests, Temperature, Functioning Day, Humidity, and Rainfall, Hour 22 are the top 5 influencing features.
7.	The model accuracy can further be increased by stacking different models as it will combine the goodness of all models.  

##	REFERENCES
•	Almabetter

•	Towards data science

•	Analytics Vidhya

•	Kaggle

•	Stack overflow

•	Python libraries technical documentation

•	CampusX on YouTube
