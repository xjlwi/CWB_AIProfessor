## TEXT PROCESSING ###
import re
import nltk as nlp
from nltk.corpus import stopwords
import wordcloud
## Create a new list
description_list = []
for description in data.dialogue:
    description = re.sub ("[^a-zA-Z]", " ", description) # match all strings that CONTAIN a NON-LETTER, convert them into blanks.
    description = description.lower() # Convert all strings to lower case.
    description = nltk.word_tokenize(description)
    description = [word for word in description if not word in set(stopwords.words("english"))] # remove stopwords
    lemma = nlp.WordNetLemmatizer() 
    description = [lemma.lemmatize(word) for word in description]
    description = " ".join(description) #This is similar to paste function in R. 
    description_list.append(description)

print(description_list[0:50])

## Add the text after processing into the previous data
data["new_script"] = description_list
temp = data.head(50)
temp.to_csv("text_postProcess.csv", index = False)

## Filter the lines by Character
luke = data[data.character=="LUKE"]
yoda = data[data.character == "YODA"]
han = data[data.character == "HAN"]
vader = data[data.character == "VADER"]

## Time to visualise the frequency of words 
wave_mask_yoda = np.array(Image.open("wordcloud_masks/yoda.png"))
wave_mask_vader = np.array(Image.open("wordcloud_masks/vader4/vader3.png"))
wave_mask_rebel = np.array(Image.open("wordcloud_masks/rebel alliance.png"))

yoda_postProcess = yoda
yoda_postProcess.to_csv("yoda_postProcess.csv", index = False)

## VISUALISATION ##
plt.subplot(figsize = (15, 15))
stopwords = set(STOPWORDS)
wordcloud = WordCloud(mask = wave_mask_vader, background_color = 'black', colormap = 'grey', contour_width = 2, contour_color = 'gray', width = 950, height = 950).generate(" ".join(vader.new_script))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis ('off')
plt.savefig('graph.png')
plt.show()


## COUNT VECTORIZER ## 
from sklearn.feature_extraction.text import CountVectorizer # bag of words

# Most used 500 words
max_features = 500
count_vectorizer = CountVectorizer(max_features = max_features, stop_words = 'english')

sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()  # x

print("mostly used {} words: {}".format(max_features,count_vectorizer.get_feature_names()))

