from langdetect import detect
from sklearn.metrics import accuracy_score
import pandas as pd

def detect_lang(tweet):
    try:
        return detect(tweet)
    except:
        return "unknown"

def count_emoticons(tweets, emoticons):
    d_count = {}
    for tweet in tweets:
        for word in tweet:
            if word in emoticons:
                if word in d_count:
                    d_count[word] += 1
                else:
                    d_count[word] = 1
    return d_count

def pos_score(scores_df, word):
	try:
		return scores_df.loc[word].pos_score.mean()
	except:
		return 0

def neg_score(scores_df, word):
    try:
        return scores_df.loc[word].neg_score.mean()
    except:
        return 0

def check_error(model, X_train, y_train, X_test, y_test):
    y_train_pred = np_utils.to_categorical(model.predict(X_train).argmax(axis=1), 3)
    y_test_pred = np_utils.to_categorical(model.predict(X_test).argmax(axis=1), 3)
    print("Accuracy on train set: {:.2}\nAccuracy on cv-set: {:.2}".format(accuracy_score(y_train, y_train_pred),
                                                accuracy_score(y_test, y_test_pred)))

def accuracy_score_per_class(y_true, y_pred, inv_d_sent):
    idx_sent = []
    acc = []
    for cl in y_true.unique():
        idx_sent.append(inv_d_sent[cl])
        acc.append((round(sum((y_true==cl)&(y_pred==cl))/sum(y_true==cl)*100,1)))
    result = pd.DataFrame({"accuracy": acc, "sentiment": idx_sent}, columns=["sentiment", "accuracy"])
    print("Macro-average accuracy: {:.1f}%".format(sum(y_true-y_pred==0.0)/len(y_true)*100))
    print("Micro-average accuracy: {:.1f}%\n#####".format(result.accuracy.mean()))
    print(result)
    return result