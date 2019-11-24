from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def sgd_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    sgd = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf',
                     SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42)),
                    ])
    sgd.fit(X_train, y_train)


    y_pred = sgd.predict(X_test)

    print('Linear SVM SGD accuracy %s' %accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred))


def naive_bayes_classifier(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    nb = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', MultinomialNB()),
                   ])
    nb.fit(X_train, y_train)

    y_pred = nb.predict(X_test)

    print('Naive Bayes accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred))


def log_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    logreg = Pipeline([('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                       ('clf', LogisticRegression(n_jobs=1, C=1e5)),
                       ])
    logreg.fit(X_train, y_train)


    y_pred = logreg.predict(X_test)

    print('Logistic Regression accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred))
