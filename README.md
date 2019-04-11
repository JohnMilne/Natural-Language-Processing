### Natural Language Processing Lab

#### This lab runs through the use of CountVectorizer, HashVectorizer and TfidfVectorizer to do Natural Language Processing on a pretrained language dataset known as fetch_20newsgroups.

#### CountVectorizer takes in a corpus and transforms the data into a numerical vector of word counts for each word in the corpus.

#### HashVectorizer uses the hashing trick to find the token string name to feature integer index mapping.

	This strategy has several advantages:

		It is very low memory scalable to large datasets as there is no need to store a vocabulary dictionary in memory
		It is fast to pickle and un-pickle as it holds no state besides the constructor parameters
		It can be used in a streaming (partial fit) or parallel pipeline as there is no state computed during fit.

	There are also a couple of cons (vs using a CountVectorizer with an in-memory vocabulary):

		There is no way to compute the inverse transform (from feature indices to string feature names) which can be a
 			problem when trying to introspect which features are most important to a model.
		There can be collisions: distinct tokens can be mapped to the same feature index. However in practice this is
			rarely an issue if n_features is large enough (e.g. 2 ** 18 for text classification problems).
		There is no IDF weighting as this would render the transformer stateful.

	The hash function employed is the signed 32-bit version of Murmurhash3.

#### TfidfVecotirizer transforms the data based on (t)ime (f)requency (i)nverse (d)ocument (f)requency in order to re-weight the
	count features into floating point values suitable for usage by a classifier.

	The formula for tf-idf is:

		tf-idf(t,d) = tf(t,d) * idf(t)

	where tf(t,d) is the term frequency, or count, for a word in a given document and idf(t) is given by the formula:

		idf(t) = log((1+n)/(1+df(t)) + 1

	where n is the total number of documents in the dataset and df(t) is how many documents a given word occurs within

	
#### Using these three text transformers, I process the fetch_20newsgroups of choice and use a GridSearched LogisticRegression to model the training data and achieved a score of 81% on the testing data.
