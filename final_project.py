import streamlit as st
import pandas as pd
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
from sklearn import linear_model, preprocessing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px

pd.DataFrame.iteritems = pd.DataFrame.items

if __name__=='__main__':

	st.title("Air Quality System")
	st.caption("https://archive.ics.uci.edu/dataset/360/air+quality")
	st.caption("If dashboard becomes unresponsive. Please try to open it in incognito or private mode.")

	st.subheader("Introductory description of dashboard:")
	st.write("Dashboard allows better decision making for air quality estimation. It is designed for \
		scientists and analysts working in domain of environmental protection for agencies like U.S. Environmental Protection Agency. \
		It enables staff to make informed decisions while understanding data better and most importantly identify potential harmful patterns. \
		To achieve this goal in this dashboard, we will use UCI Air Quality data to demonstrate key insights and forecasts.")

	path = 'AirQualityUCI_processed.csv'
	months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
				'August', 'September', 'October', 'November', 'December']

	models_options = ['Random Forest','Gradient Boosting',
						'Fastest','Most Accurate']

	model_type = st.sidebar.selectbox('Select method for forecasting: ', models_options)
	if model_type == 'Random Forest':
		forecaster = ForecasterAutoreg(
	                regressor = RandomForestRegressor(random_state=123),
	                lags = 24
	             )

		forecaster_lt = ForecasterAutoreg(
	                regressor = RandomForestRegressor(random_state=123),
	                lags = 6
	             )
	elif model_type == 'Fastest':
		
		forecaster = ForecasterAutoreg(
	                regressor = linear_model.LinearRegression(),
	                lags = 24
	             )

		forecaster_lt = ForecasterAutoreg(
	                regressor = linear_model.LinearRegression(),
	                lags = 6
	             )
	elif model_type == 'Most Accurate':

		forecaster = ForecasterAutoreg(
	                regressor = GradientBoostingRegressor(),
	                lags = 24
	             )

		forecaster_lt = ForecasterAutoreg(
	                regressor = GradientBoostingRegressor(),
	                lags = 6
	             )

	df = pd.read_csv(path, sep=';', engine='python')
	df.dropna(how='all',axis=0, inplace=True)

	feature_options = df.columns[2:]
	feature_type = st.sidebar.selectbox("Select feature for study: ", feature_options)
	
	df['Month'] = df['Date']
	df['Month'] = df['Month'].apply(lambda x: months[int(float(str(x).split('/')[1])) - 1])
	df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
	
	df[feature_type] = df[feature_type][df[feature_type] != -200] 
	df[feature_type][df[feature_type].isna()] = df[feature_type].mean()
	
	
	# Trend

	st.subheader("Quarterly trend of air quality")
	st.write("Proportion and trend of constituents of air show if there exists are right air quality or not.")

	tab1, tab2, tab3, tab4 = st.tabs(["1st quarter trend (2004)", "2nd quarter trend (2004)", "3rd quarter trend (2004)", "4th quarter trend (2004)"])
	grouped_data = df.groupby('Month')[feature_type].mean().reset_index()
	
	with tab1:
		
		plt.clf()
		plot = sns.lineplot(data=grouped_data.iloc[[4,3,7]], x ='Month', y=feature_type)
		st.pyplot(plot.get_figure())
		
	with tab2:

		plt.clf()
		plot = sns.lineplot(data=grouped_data.iloc[[0, 8, 6]], x ='Month', y=feature_type)
		st.pyplot(plot.get_figure())

	with tab3:
		
		plt.clf()
		plot = sns.lineplot(data=grouped_data.iloc[[5, 1, 11]], x ='Month', y=feature_type)
		st.pyplot(plot.get_figure())
		
	with tab4:

		plt.clf()
		plot = sns.lineplot(data=grouped_data.iloc[[10, 9, 2]], x ='Month', y=feature_type)
		st.pyplot(plot.get_figure())
	

	# Shorterm forecast
	
	st.subheader("Short-term Forecast")

	st.write("Short-term forecast allows us to plan for rapid weather and environment changes.")

	tab1, tab2, tab3 = st.tabs(["24 Hrs forecast", "48 Hrs forecast", "72 Hrs forecast"])

	window = st.sidebar.slider('Number of points to be considered for short-term forecast', 50, 5000, 1000)
	
	forecaster.fit(y=df[feature_type].iloc[-window:])

	
	with tab1:
		
		steps = 25
		predictions = forecaster.predict(steps=steps)
		
		tmp_df = pd.DataFrame({}, columns=['date', 'preds'])
		tmp_df['Forecast'] = predictions
		tmp_df['Time'] = pd.date_range(start='4/4/2005', end='4/5/2005', freq='H')
		
		st.line_chart(data=tmp_df, x ='Time', y='Forecast')

	with tab2:

		steps = 49
		predictions = forecaster.predict(steps=steps)
		
		tmp_df = pd.DataFrame({}, columns=['date', 'preds'])
		tmp_df['Forecast'] = predictions
		tmp_df['Time'] = pd.date_range(start='4/4/2005', end='4/6/2005', freq='H')
		
		st.line_chart(data=tmp_df, x ='Time', y='Forecast')

	with tab3:

		steps = 73
		predictions = forecaster.predict(steps=steps)
		
		tmp_df = pd.DataFrame({}, columns=['date', 'preds'])
		tmp_df['Forecast'] = predictions
		tmp_df['Time'] = pd.date_range(start='4/4/2005', end='4/7/2005', freq='H')
		
		st.line_chart(data=tmp_df, x ='Time', y='Forecast')


	# Longterm Forecast

	st.subheader("Long-term Forecast")
	st.write("Long-term forecasts show long lasting effects and trends of different gases and constituents present in Air.")

	forecaster_lt.fit(y=grouped_data[feature_type][[4,3,7,0, 8, 6, 5, 1, 11, 10, 9, 2]])

	tab1, tab2, tab3 = st.tabs(["3 months forecast", "6 months forecast", "1 year forecast"])
	
	with tab1:
		
		steps = 3     
		predictions = forecaster_lt.predict(steps=steps)
		
		tmp_df = pd.DataFrame({}, columns=['date', 'preds'])
		tmp_df['Forecast'] = predictions
		tmp_df['Time'] = pd.date_range(start='4/4/2005', end='7/4/2005', freq='M')
		
		st.line_chart(data=tmp_df, x ='Time', y='Forecast')

	with tab2:

		steps = 6
		predictions = forecaster_lt.predict(steps=steps)
		
		tmp_df = pd.DataFrame({}, columns=['date', 'preds'])
		tmp_df['Forecast'] = predictions
		tmp_df['Time'] = pd.date_range(start='4/4/2005', end='10/6/2005', freq='M')
		
		st.line_chart(data=tmp_df, x ='Time', y='Forecast')

	with tab3:

		steps = 12
		predictions = forecaster_lt.predict(steps=steps)
		
		tmp_df = pd.DataFrame({}, columns=['date', 'preds'])
		tmp_df['Forecast'] = predictions
		tmp_df['Time'] = pd.date_range(start='4/4/2005', end='4/7/2006', freq='M')
		
		st.line_chart(data=tmp_df, x ='Time', y='Forecast')

	st.subheader("Principal Components Analysis")
	st.write("Principtal components analysis allows us to track patterns with high variance and it also allows us to visualize high dimensional data.")

	hardcoded_window = 1000
	
	n_pcs = st.sidebar.slider('Number of principal components to be displayed', 2, 4, 3)
	pca = PCA(n_components = n_pcs)


	df_array = df.iloc[:, 2:14].to_numpy()
	trans_arr = pca.fit_transform(preprocessing.normalize(df_array))
	
	if n_pcs == 3:
		fig = px.scatter_3d(trans_arr, x=trans_arr[:hardcoded_window, 0], y=trans_arr[:hardcoded_window, 1], z=trans_arr[:hardcoded_window,2])
		st.plotly_chart(fig)

	elif n_pcs == 4:
		fig = px.scatter_3d(trans_arr, x=trans_arr[:hardcoded_window, 0], y=trans_arr[:hardcoded_window, 1], z=trans_arr[:hardcoded_window,2], color=trans_arr[:hardcoded_window,3])
		st.plotly_chart(fig)
	elif n_pcs == 2:
		fig = px.scatter(x=trans_arr[:hardcoded_window, 0], y=trans_arr[:hardcoded_window, 1])
		st.plotly_chart(fig)

	st.subheader("Cluster Analysis")

	st.write("Cluster Analysis shows structure of principcal components of data and details about data clusters.")

	n_clusters = st.sidebar.slider("Number of clusters for cluster analysis: (Dependent on PCA) ", 2, 15, 6)
	
	auto_cluster = st.sidebar.checkbox("Generate plot by Elbow Method.(Time consuming process)")

	if auto_cluster:
		K = range(2, 15, 2)
		fits = []
		score = []


		for k in K:
			# train the model for current value of k on training data
			model = KMeans(n_clusters = k, random_state = 0, n_init='auto').fit(trans_arr)

			# append the model to fits
			fits.append(model)

			# Append the silhouette score to scores
			score.append(silhouette_score(trans_arr, model.labels_, metric='euclidean'))
		plt.clf()
		fig4 = sns.lineplot(x = K, y = score)
		st.pyplot(fig4.get_figure())


	kmean_clf = KMeans(n_clusters)
	trans_k_arr = kmean_clf.fit_transform(trans_arr)
	
	kmeans_ys = st.sidebar.selectbox("Choose Y-axis feature for K-means analysis: ", feature_options)

	plt.clf()
	fig3 = sns.boxplot(x = kmean_clf.labels_, y = df[kmeans_ys])
	st.pyplot(fig3.get_figure())

	st.caption("Clustered visualization of principal components.")
	if n_pcs == 2:
		fig2 = px.scatter(x=trans_arr[:hardcoded_window, 0], y=trans_arr[:hardcoded_window, 1], color=kmean_clf.labels_[:hardcoded_window])
		st.plotly_chart(fig2)	

	elif n_pcs == 3 or n_pcs == 4:
		
		fig2 = px.scatter_3d(trans_arr, x=trans_arr[:hardcoded_window, 0], y=trans_arr[:hardcoded_window, 1], z=trans_arr[:hardcoded_window,2], color=kmean_clf.labels_[:hardcoded_window])
		st.plotly_chart(fig2)
	else:
		st.caption("Higher dimensions cannot be displayed!")
