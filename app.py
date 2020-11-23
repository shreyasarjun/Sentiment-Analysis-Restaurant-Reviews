# Importing essential libraries
from flask import Flask, render_template, request
import pickle

#pre processing models
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
cv = pickle.load(open('cv-transform.pkl','rb'))
classifier = pickle.load(open('restaurant-sentiment-mnb-model.pkl', 'rb'))



app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
    	ps=PorterStemmer()
    	message = request.form['message']
    	message = re.sub('[^a-zA-Z]',' ',message).lower().split()
    	review_words = [word for word in message if not word in set(stopwords.words('english'))-{'not'}]
    	review = [ps.stem(word) for word in review_words]
    	review = ' '.join(review)
    	data=[review]
    	vect = cv.transform(data).toarray()
    	my_prediction = classifier.predict(vect)
    	return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
