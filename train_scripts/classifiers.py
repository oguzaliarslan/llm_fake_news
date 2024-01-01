import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
from tqdm import tqdm
from joblib import dump
import argparse
import pandas as pd

def call_models():
   # tree-based: Gradientboosting, AdaBoost, Random Forest, Extra Trees, Decision Tree, CatBoost, LightGBM, XGBoost
   # others: lr, nv, pac, rc
    return  {
        'Logistic Regression': (LogisticRegression(), {}),
        #'Ridge Classifier': (RidgeClassifier(), {}),
        'Passive Aggressive Classifier': (PassiveAggressiveClassifier(), {}),
        'Naive Bayes': (MultinomialNB(), {}),
        'Gradient Boosting': (GradientBoostingClassifier(), {}),
        'AdaBoost': (AdaBoostClassifier(), {}),
        'Random Forest': (RandomForestClassifier(), {}),
        'Extra Trees': (ExtraTreesClassifier(), {}),
        'Decision Tree': (DecisionTreeClassifier(), {}),
        #'Bagging': (BaggingClassifier(), {}),
        'LGBM': (LGBMClassifier(), {}),
        'CatBoost': (CatBoostClassifier(verbose=False), {}),
        'XGBoost': (xgb.XGBClassifier(tree_method='gpu_hist'), {})
    }


def text_classification(texts, labels) -> pd.DataFrame:
    """
    Perform text classification using different feature extraction methods and models.
    Args:
        texts (pd.DataFrame): A series of text samples.
        labels (pd.DataFrame): A series of corresponding labels for the text samples.
    Returns:
        pd.DataFrame
    """
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.4, random_state=42)

    feature_extraction_methods = {
        'Bag of Words': CountVectorizer(),
        'TF-IDF': TfidfVectorizer()
        #'Hashing Vectorizer': HashingVectorizer(n_features=500)
        #'Word2Vec': Word2Vec()  # This creates lots of problem..
    }

    models = call_models()
    results = []

    for model_name, (model, _) in tqdm(models.items(), total=len(models)):
        model_folder = f"./models_no_gridsearch_2/{model_name.replace(' ', '_')}_results"
        os.makedirs(model_folder, exist_ok=True)
        for fe_name, fe in tqdm(feature_extraction_methods.items(), total=len(feature_extraction_methods)):
            try:
                print(f'Sanity check: {model_name, fe_name}')
                X_train_fe = fe.fit_transform(X_train)
                X_test_fe = fe.transform(X_test)
                
                print('Training')
                model.fit(X_train_fe, y_train)
                
                print('Predictions')
                predictions = model.predict(X_test_fe)
                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions, average='weighted')
                recall = recall_score(y_test, predictions, average='weighted')
                f1 = f1_score(y_test, predictions, average='weighted')

                print(f"Accuracy: {accuracy}")
                row = [model_name, fe_name, accuracy, precision, recall, f1, {}]
                results.append(row)

                model_filename = f"{fe_name.replace(' ', '_')}_model.joblib"
                model_filepath = os.path.join(model_folder, model_filename)
                dump(model, model_filepath)

                vectorizer_filename = f"{fe_name.replace(' ', '_')}_vectorizer.joblib"
                vectorizer_filepath = os.path.join(model_folder, vectorizer_filename)
                dump(fe, vectorizer_filepath)

                csv_filename = f"{fe_name.replace(' ', '_')}_results.csv"
                csv_filepath = os.path.join(model_folder, csv_filename)
                model_df = pd.DataFrame(results, columns=['Model', 'Feature Extraction', 'Accuracy', 'Precision','Recall','F1','Best Params',])
                model_df.to_csv(csv_filepath, index=False)

            except Exception as e:
                print(f"Error occurred for model '{model_name}' with feature extraction method '{fe_name}': {str(e)}")
            

    headers = ['Model', 'Feature Extraction', 'Accuracy', 'Precision','Recall','F1','Best Params']
    model_df = pd.DataFrame(results, columns=headers)
    return model_df




def call_models_grid():
    tolerance = 1e-2
    
    return  {
         'Logistic Regression': (LogisticRegression(), {'C': [0.1, 1, 10], 'tol': [tolerance]}),
         'Passive Aggressive Classifier': (PassiveAggressiveClassifier(), {'C': [0.1, 1, 10], 'tol': [tolerance]}),
         'Naive Bayes': (MultinomialNB(), {}),
         'Random Forest': (RandomForestClassifier(), {'n_estimators': [50, 100, 200], 'max_depth':[5,10,20]}),
         'Gradient Boosting': (GradientBoostingClassifier(), {'learning_rate': [0.1, 0.01], 'n_estimators': [50, 100], 'tol':[tolerance]}),
         'AdaBoost': (AdaBoostClassifier(), {'learning_rate': [0.1, 0.01], 'n_estimators': [50, 100]}),
        'Extra Trees': (ExtraTreesClassifier(), {'n_estimators': [25, 50], 'max_depth':[5,10]}),
        'Decision Tree': (DecisionTreeClassifier(), {'max_depth': [5, 10]}),
        'LGBM': (LGBMClassifier(), {'learning_rate': [0.1, 0.01], 'n_estimators': [50, 100], 'max_depth': [5, 10]}),
        'CatBoost': (CatBoostClassifier(verbose=False), {'learning_rate': [0.1, 0.01], 'n_estimators': [50, 100]}),
    }

def text_classification_grid(texts, labels, out) -> pd.DataFrame:
    """
    Perform text classification using different feature extraction methods and models.
    Args:
        texts (pd.DataFrame): A series of text samples.
        labels (pd.DataFrame): A series of corresponding labels for the text samples.
    Returns:
        pd.DataFrame
    """
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.4, random_state=42)

    feature_extraction_methods = {
        'Bag of Words': CountVectorizer(),
        'TF-IDF': TfidfVectorizer()
        #'Hashing Vectorizer': HashingVectorizer(n_features=500)
        #'Word2Vec': Word2Vec()  # This creates lots of problem..
    }

    models = call_models_grid()
    results = []

    for model_name, (model, param_grid) in tqdm(models.items(), total=len(models.items())):
        model_folder = f"{out}/{model_name.replace(' ', '_')}_results"
        os.makedirs(model_folder, exist_ok=True)
        for fe_name, fe in tqdm(feature_extraction_methods.items(), total=len(feature_extraction_methods)):
            try:
                print(f'Sanity check: {model_name, fe_name}')
                X_train_fe = fe.fit_transform(X_train)
                X_test_fe = fe.transform(X_test)

                print('Grid Search')
                grid_search = GridSearchCV(model, param_grid, cv=3)

                print('Training')
                grid_search.fit(X_train_fe, y_train)
                
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_

                print('Predictions')
                predictions = best_model.predict(X_test_fe)
                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions, average='weighted')
                recall = recall_score(y_test, predictions, average='weighted')
                f1 = f1_score(y_test, predictions, average='weighted')

                print(f"Accuracy: {accuracy}")
                row = [model_name, fe_name, accuracy, precision, recall, f1, best_params]
                results.append(row)

                model_filename = f"{fe_name.replace(' ', '_')}_model.joblib"
                model_filepath = os.path.join(model_folder, model_filename)
                dump(model, model_filepath)

                vectorizer_filename = f"{fe_name.replace(' ', '_')}_vectorizer.joblib"
                vectorizer_filepath = os.path.join(model_folder, vectorizer_filename)
                dump(fe, vectorizer_filepath)

                csv_filename = f"{fe_name.replace(' ', '_')}_results.csv"
                csv_filepath = os.path.join(model_folder, csv_filename)
                model_df = pd.DataFrame(results, columns=['Model', 'Feature Extraction', 'Accuracy', 'Precision','Recall','F1','Best Params',])
                model_df.to_csv(csv_filepath, index=False)

            except Exception as e:
                print(f"Error occurred for model '{model_name}' with feature extraction method '{fe_name}': {str(e)}")
            

    headers = ['Model', 'Feature Extraction', 'Accuracy', 'Precision','Recall','F1','Best Params']
    model_df = pd.DataFrame(results, columns=headers)
    return (model_df)



def main():
    parser = argparse.ArgumentParser(description='Text Classification with CLI')
    parser.add_argument('--input_data', type=str, help='Path to input data file (CSV)')
    parser.add_argument('--output_folder', type=str, help='Path to the output folder to store results')
    parser.add_argument('--grid_search', action='store_true', help='Whether to perform grid search or not')
    print("qwe")
    args = parser.parse_args()

    input_data = pd.read_csv(args.input_data)
    try:
        texts = input_data['clean_text']
    except:
        texts = input_data['text']
    labels = input_data['label']
    texts.fillna('', inplace=True)
    output_folder = args.output_folder

    if args.grid_search:
        result = text_classification_grid(texts, labels, output_folder)
    else:
        result = text_classification(texts, labels)

    result.to_csv(f'{output_folder}/text_classification_results.csv', index=False)
    
    
if __name__ == "__main__":
    main()