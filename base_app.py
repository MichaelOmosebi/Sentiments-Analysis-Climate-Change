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
from PIL import Image

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("tfidf_vector.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("train.csv")

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
	options = ["Information", "EDA", "Prediction", "Model Explanation", "App Developers"]
	selection = st.sidebar.selectbox("Navigation", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("A Quick Look at the Dataset")
		# You can read a markdown file from supporting resources folder
		st.markdown("See how the data is transformed here 🔎")

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

		st.markdown("WORDCLOUD IMAGES SHOWING MOST MENTIONED WORDS & SENTIMENT TWEETS")
		# Create some sample text
		text = " ".join(raw['message'])

		# Create and generate a word cloud image for each sentiment class:

		gb = raw.groupby('sentiment')
		Anti = "".join(gb.get_group(-1)['message'])
		Neutral_tweets= "".join(gb.get_group(0)['message'])
		Pro = "".join(gb.get_group(1)['message'])
		News = "".join(gb.get_group(2)['message'])

		'Some Anti Climate Change Tweets'
		Anti_words = gb.get_group(-1)['message']
		Anti_words[0:6]

		'Most Frequesnt Words in Anti Climate Change Tweets'
		wc = WordCloud(background_color='black')
		img = wc.generate(Anti)
		plt.figure(figsize=(6,6))
		plt.imshow(img)
		plt.axis('off')
		plt.show()
		st.pyplot()

		'Some Neutral-to-Climate Change Tweets'
		Neutral = gb.get_group(0)['message']
		Neutral[0:6]

		'Most Frequesnt Words in Neutral to Climate Change Tweets'
		wc = WordCloud(background_color='black')
		img = wc.generate(Neutral_tweets)
		plt.figure(figsize=(6,6))
		plt.imshow(img)
		plt.axis('off')
		plt.show()
		st.pyplot()

		'Some Pro Climate Change Tweets'
		Pro_tweets = gb.get_group(1)['message']
		Pro_tweets[0:6]

		'Most Frequesnt Words in Pro Climate Change Tweets'
		wc = WordCloud(background_color='black')
		img = wc.generate(Pro)
		plt.figure(figsize=(6,6))
		plt.imshow(img)
		plt.axis('off')
		plt.show()
		st.pyplot()

		'Some News Tweets'
		News_tweets = gb.get_group(2)['message']
		News_tweets[0:6]

		'Most Frequesnt Words in News Tweets'
		wc = WordCloud(background_color='black')
		img = wc.generate(News)
		plt.figure(figsize=(6,6))
		plt.imshow(img)
		plt.axis('off')
		plt.show()
		st.pyplot()

		wordcloud = WordCloud().generate(text)

		'>>>The Most repeated words in the tweet messages are: ``Climate, Change, Global, Warming, Change, https, RT, Today, CO e.t.c.``'

	# Building out the predication page
	if selection == "Prediction":
		#st.info("Classify with: {}".format(model_choice))
		# Creating a text box for user input
		#tweet_text = st.text_area("Enter Text","Type Here")
		models = ["Logistic Regression", "Random Forest", "Multinomial Naive Bayes", "Support Vector Machine", "K Nearest Neighbours"]
		model_choice = st.sidebar.selectbox("Choose Your Model", models)
		st.info("Make a classification using our {} model".format(model_choice))
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			if model_choice == 'Logistic Regression':
				#st.subheader("Classify with: {}".format(model_choice))
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

	if selection == "Model Explanation":
		st.info('An Explanation of our Top Performing Model - The Logistic Regression:')
		st.write("The Logistic Regression model, makes use of a logistic function: f(x), that takes the shape of a S-Shape Curve known as SIGMOID, the logistic regression model takes the training data and maps the independent variables (features) to exist only between the multi class dependent variables (target) which are (Anti:-1), (Neutral:0), (Pro:1), (News:2). It selects a reference class and uses the one vs rest (OvR) algorithm to predict the probability of a tweet belonging to a particular dependent or target class.")

    ##contact page
	if selection == 'App Developers':
		st.info('Contact details in case you have any query or would like to know more of our designs: 🧑🏻‍🤝‍🧑🏽🤝')
		st.write('Sodiq: sodiq@smend.com')
		st.write('Dorcas: dorcas@smend.com')
		st.write('Eteng: eteng@smend.com')
		st.write('Michael: michael@smend.com')
		st.write('Nichodemus: nichodemus@smend.com')

        # Footer
		image = Image.open('resources/Logo.jpg')
		st.image(image, caption='Team2(2110ACDS_T2)', use_column_width=True)

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
