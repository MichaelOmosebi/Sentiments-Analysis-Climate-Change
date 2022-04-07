"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("tfidf_vector.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("train.csv")

#Load other tables for EDA
#info = pd.read_csv("tables/info.csv")
#comparison = pd.read_csv('comparison.csv')

cleaned_data = pd.read_csv("preprocessed_data.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information", "EDA", "Model Explanation", "App Developers"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some how we transformed the data here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

		st.subheader("Cleaned Twitter data and label")
		if st.checkbox('Show clean data'): # data is hidden if box is unchecked
			st.write(cleaned_data) # will write the df to the page

	# Building out the predication page
	if selection == "EDA":
		st.info("This will be an exploration and explanation of the dataset used and insights derived respectively")
		# Creating EDA table views
		# Describe
		st.markdown("A Statistical Description of the Dataset")
		st.table(raw.groupby('sentiment').describe(include=['object']))

		st.markdown("A COUNTPLOT OF SENTIMENTS CLASS IN THE TWEETS")
		#st.countplot(x = 'sentiment', data = train)
		fig = plt.figure(figsize=(10, 4))
		#ax.set(xlabel='Tweet Count', ylabel='Sentiment Class')
		sns.countplot(x = 'sentiment', data = raw)
		st.pyplot(fig)
		'>>>We have an imbalanced Dataset classification --- building a model with this raw dataset set makes our prediction bias towards the weightier class (1)'

		st.markdown("A WORDCLOUD IMAGE SHOWING MOST MENTIONED WORDS")
		# Create some sample text
		text = " ".join(raw['message'])

		# Create and generate a word cloud image:
		wordcloud = WordCloud().generate(text)

		# Display the generated image:
		plt.imshow(wordcloud, interpolation='bilinear')
		plt.axis("off")
		plt.show()
		st.pyplot()
		#Cater for deprecation errors
		st.set_option('deprecation.showPyplotGlobalUse', False)
		'>>>The Most repeated words in the tweet message are: ``Climate, Change, Global, Warming, Change, https, RT, Today, CO e.t.c.``'

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")
		models = ["Logistic Regression", "Random Forest", "Multinomial Naive Bayes", "Support Vector Machine", "K Nearest Neighbours"]
		model_choice = st.sidebar.selectbox("Choose Option", models)

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			if model_choice == 'Logistic Regression':
				predictor = joblib.load(open(os.path.join("model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
            # st.write(prediction)
			elif model_choice == 'Random Forest':
				predictor = joblib.load(open(os.path.join("RFC.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
            # st.write(prediction)
			elif model_choice == 'Multinomial Naive Bayes':
				predictor = joblib.load(open(os.path.join("MNB.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
            # st.write(prediction)
			elif model_choice == 'SupportVectorMachine':
				predictor = joblib.load(open(os.path.join("SVM.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
			
			elif model_choice == 'K Nearest Neighbours':
				predictor = joblib.load(open(os.path.join("KNN.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

    ##contact page
	if selection == 'App Developers':
		st.info('Contact details in case you any query or would like to know more of our designs:')
		st.write('Sodiq: sodiq@smend.com')
		st.write('Dorcas: dorcas@smend.com')
		st.write('Eteng: eteng@smend.com')
		st.write('Michael: michael@smend.com')
		st.write('Nichodemus: nichodemus@smend.com')

        # Footer 
        #image = Image.open('resources/imgs/EDSA_logo.png')

        #st.image(image, caption='Team2(2110ACDS_T2)', use_column_width=True)

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
