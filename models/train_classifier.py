import sys
import pandas as pd
import pickle
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from sqlalchemy import create_engine
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

def load_data(database_filepath):
    """ Load data from the database
    
    Args:
        database_filepath(string): database filepath input in the interpreter
        
    Returns:
        X(numpy.ndarray): values of message column
        Y(numpy.ndarray): values of 36 target columns
        category_names(pandas.core.indexes.base.Index): columns' name of 36 target columns
    """
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(con=engine, table_name='DisasterResponse')
    X = df['message'].values
    Y = df.loc[:, 'related':'direct_report'].values
    category_names = df.columns[4:]
    
    return X, Y, category_names


def tokenize(text):
    """
    Function to return tokenized text in order to build model
    """
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens

def build_model():
    """
    Builds a pipeline to build a classification model
    The pipeline consists of:
        1. CountVectorizer
        2. TfidTransformer
        3. MultioutputClassifier
        4. RandomForestClassifier
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    
    params = {
        'clf__estimator__n_estimators': [100, 200],
        'clf__estimator__min_samples_leaf': [1, 8],
        'clf__estimator__n_jobs': [None, -1]
    }

    cv = GridSearchCV(estimator=pipeline, param_grid=params, cv=3, n_jobs=-1)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model with the test data and predict data based on Precision, Recall, and F1-Score feature
    """
    
    df_y_test = pd.DataFrame(data=Y_test, columns=category_names)
    
    y_pred = model.predict(X_test)
    df_y_pred = pd.DataFrame(data=y_pred, columns=category_names)
    
    print(classification_report(df_y_test, df_y_pred))

def save_model(model, model_filepath):
    """
    Save model as pkl file to apply to Webapp
    """
    
    with open(model_filepath, 'wb') as model_file:
        pickle.dump(model,model_file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()