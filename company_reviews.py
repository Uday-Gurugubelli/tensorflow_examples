import pandas as pd

# read data
train_df = pd.read_csv("./NLP_Data/train.csv")
test_df = pd.read_csv("./NLP_Data/test.csv")
print(train_df.columns)
label = train_df.overall
drop_list = ['ID', 'Place', 'location', 'date', 'status', 'job_title']
train_df.drop(drop_list, axis=1, inplace=True)
test_df.drop(drop_list, axis=1, inplace=True)

train_df = train_df.apply(lambda x: x.replace("NaN", " "))

#print(train_df["score_1"].isnull().values.any())
train_df["summary"] = train_df["summary"].fillna(" ")
train_df["positives"] = train_df["positives"].fillna(" ")
train_df["negatives"] = train_df["negatives"].fillna(" ")
train_df["advice_to_mgmt"] = train_df["advice_to_mgmt"].fillna(" ")

test_df["summary"] = test_df["summary"].fillna(" ")
test_df["positives"] = test_df["positives"].fillna(" ")
test_df["negatives"] = test_df["negatives"].fillna(" ")
test_df["advice_to_mgmt"] = test_df["advice_to_mgmt"].fillna(" ")

train_df = train_df.fillna(0)
test_df = test_df.fillna(0)
print(train_df.describe())
print(test_df.describe())
#reviews_df = reviews_df.sample(frac = 0.1, replace = False, random_state=42)
# append the positive and negative text reviews
train_df["review"] = train_df["summary"] + train_df["positives"]+train_df["negatives"]+train_df["advice_to_mgmt"]
test_df["review"] = test_df["summary"] + test_df["positives"]+test_df["negatives"]+test_df["advice_to_mgmt"]
print(train_df.head(25))
print(test_df.head(25))

from nltk.corpus import wordnet

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)

# clean text data
train_df["review_clean"] = train_df["review"].apply(lambda x: clean_text(x))
test_df["review_clean"] = test_df["review"].apply(lambda x: clean_text(x))
print(train_df.head())

# add number of characters column
train_df["nb_chars"] = train_df["review"].apply(lambda x: len(x))
test_df["nb_chars"] = test_df["review"].apply(lambda x: len(x))
# add number of words column
train_df["nb_words"] = train_df["review"].apply(lambda x: len(x.split(" ")))
test_df["nb_words"] = test_df["review"].apply(lambda x: len(x.split(" ")))
print(train_df)
print(test_df)

# create doc2vec vector columns
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

df = train_df["review_clean"] + test_df["review_clean"]
df = df.fillna(" ")
print(df)
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(df.apply(lambda x: x.split(" ")))]

# train a Doc2Vec model with our text data
model = Doc2Vec(documents, vector_size=100, window=2, min_count=1, workers=4)

# transform each document into a vector data
doc2vec_df = train_df["review_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
train_dff = pd.concat([train_df, doc2vec_df], axis=1)
print(train_dff.columns)


doc2vec_df = test_df["review_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
test_dff = pd.concat([test_df, doc2vec_df], axis=1)
print(test_dff.columns)


label = "overall"
ignore_cols = [label, "summary", "positives", "negatives", "advice_to_mgmt", "review", "review_clean"]
features = [c for c in train_dff.columns if c not in ignore_cols]

# split the data into train and test
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_dff[features], train_dff[label], test_size = 0.20, random_state = 42)

rf = RandomForestClassifier(n_estimators = 300, random_state = 42)
rf.fit(X_train, y_train)


from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

y_pred = [x[1] for x in rf.predict_proba(X_test)]
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label = 1)

roc_auc = auc(fpr, tpr)

plt.figure(1, figsize = (15, 10))
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

test_df = pd.read_csv("./NLP_Data/test.csv")
print(test_df.ID.values)
result_df = pd.DataFrame({'ID':test_df.ID.values, 'overall':rf.predict(test_dff[features])}) #], columns=['ID', 'overall'])
result_df.to_csv("company_submission.csv", index=False)
