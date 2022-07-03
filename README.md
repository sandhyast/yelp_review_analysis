
# Sentiment Analysis of Yelp Reviews <br>
*by Ashlyn B Aske, Sandhya Thomas & Nikolas Anderson*

**Introduction**

As a popular website for reviewing restaurants, stores, and other businesses, Yelp has taken its place as a commonly used platform useful for helping customers make decisions based on previous experiences of others. Along with the ability to leave reviews for specific businesses, Yelp’s five-star rating scale helps others determine whether places leave customers with either positive or negative experiences. For our project, we wanted to see whether we could create a neural network that is able to predict the sentiment of Yelp reviews as either positive or negative. While our goal was to get an introduction to the use of natural language processing with neural networks, we also considered that as a broader application of our project, businesses may utilize networks to further analyze their performance and customer satisfaction.

**Data Preprocessing and Tokenization**

For our models, we used a dataset of 10,000 Yelp reviews and their ratings from Kaggle (https://www.kaggle.com/datasets/omkarsabnis/yelp-reviews-dataset?resource=download). After removing all unnecessary columns from the dataset except the ratings and the reviews themselves, we used a package of common and likely to be irrelevant words (stopwords) from scikit-learn, and dropped all stopwords from the review dataset. Before being able to use the reviews in a network, we also had to encode each word within the review to a number that the network could take in. Using the Tokenizer class from Keras, we were able to easily complete this. We decided to look at the 500 most common words within our dataset of 10,000 reviews and use the Tokenizer to create a word index mapping those words to a number. Then, each word in every review was either mapped to the number indexed to the word or the unknown token. We also had to consider the fact that each review was a different length. After creating a word count distribution to maximize the number of words we were able to include for each review, we decided to set this to 200. If reviews were longer than 200 words, they were truncated at the end. If they were shorter than 200 words, padding characters were added to the review at the end.

**Design One: Predicting a Review’s Rating**

Initially, we attempted to create a model that, given a review, would predict the corresponding 5-point rating. To do this, we one-hot encoded each review in our training dataset.

We then fed the encoded reviews through a model using an LSTM layer, dropout and regularization to manage overfitting, and a ReLU activation function.


To evaluate our model, we looked at the mean-squared error (mse) for the test and training data. Unfortunately, this model did not work well on the test data. Even with several different measures to prevent overfitting, we were not able to achieve a mean-squared error under 1.

We believe this may be because there is not a strong universal understanding across Yelp reviewers of what type of service constitutes a 1 vs. a 2, or a 4 vs. a 5, for instance. This makes it hard for our neural network to learn a pattern between the review’s text and a specific number rating, because the reviews might not contain such a pattern.

**Design Two: Positive vs. Negative Sentiment model**

With this issue in mind, we deviated from the rating prediction goal and created another model that would classify reviews as either generally positive or generally negative. The formatting of the data and most of the layers in the model were the same as in our original design, but instead of using a ReLU activation function and generating a linear distribution, we modified the last layer to do a binary classification using a sigmoid activation function.


We evaluated the model using the binary cross entropy error function and achieved a much better result.

**Design Three: Unordered Input**

Our third and final design was simply an unordered version of our previous model. We hoped that this method, despite being less complex, could be promising due to the smaller size of the input. In this case, the input was 500xn rather than 200x500 where n was the maximum count of a word in a review. More specifically, each review was transformed into a 500 digit long sequence such that each digit represented the count of a specific word. The parameters limiting the maximum size of the review and number of words to use was still in effect. Our theory was that the number of times certain relevant words appeared, such as “good” or “bad,” would be enough to tell the difference between positive and negative words. We used a sequential model with 2 dense layers and dropout.

We only managed to get 57% training accuracy and 59% testing accuracy with binary classification. Our conclusion from the third design was that the appearance of certain words alone is not enough to capture the sentiment of text because for example “not good” should indicate a negative sentiment despite including the word “good.” Although some negative reviews might be easy to identify with this method due to the inclusion of expletives, that accounts for only a small subset of the data.

**Conclusion & next steps**

One aspect which we had originally not focused on but believe could make a difference would be focusing more on the use of stop words. The problem with using a module to eliminate stopwords is that certain vocabulary might be common across all reviews but how they are used might be important to capturing the sentiment. An example would be the word “very” which could come in front of the word “good” more often in higher ratings. Another factor that might have affected our accuracy was the nature of the yelp review ecosystem itself. Different types of businesses have different rating distributions, reviewers have different standards for assigning ratings, and there might even be different standards geographically. One potential next step would be to look at pairs or larger combinations of words as inputs as well as creating a custom list of included words so that we can keep low input size for the module, retain order when it matters, and ensure any relevant vocabulary is not lost.
