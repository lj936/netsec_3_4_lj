import json
import pandas as pd
from my_feature_encoder import My_Feature_Encoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from preprocess import preprocess

def extract_string(value):
    return value[0] if isinstance(value, list) and len(value) == 1 else value

def create_json_results(y_lab_enc, predicted_results, ids_test, level):
    prediction = y_lab_enc.inverse_transform(predicted_results)
    df = pd.DataFrame({'id': ids_test, 'prediction': prediction})
    result_dict = {}
    for index, row in df.iterrows():
        result_dict[str(row['id'])] = str(row['prediction'])
    json_filename = 'results_' + level + '.json'
    with open(json_filename, 'w') as json_file:
        json.dump(result_dict, json_file, indent=4)
    print(f"🛈 {json_filename} successfully created 🛈")

#for each fine-grain level, the data is processed, a random forest is fitted and prediction is made based on test data
for level in ['toplevel_1', 'midlevel_2', 'finegrained_3']:

    print(f"🛈 Actual Level: {level} 🛈")

    print("Processing the data")

    #Creating Trainingsdataframe
    with open('x_train.json', 'r') as file:
        x_train_points = [json.loads(line) for line in file]
    with open('y_train_' + level + '.json', 'r') as file:
        y_train_points = json.load(file)
    train_df = pd.DataFrame(x_train_points)
    train_df['label'] = train_df['id'].astype(str).map(y_train_points)

    #Creating Testdataframe
    with open('x_test.json', 'r') as file:
        x_test_points = [json.loads(line) for line in file]
    with open('y_test_' + level + '.json', 'r') as file:
        y_test_points = json.load(file)
    test_df = pd.DataFrame(x_test_points)
    test_df['label'] = test_df['id'].astype(str).map(y_test_points)

    #Creating Labelencoder
    y_lab_enc = LabelEncoder()
    y_lab_enc.fit(train_df['label'])

    #The feature 'dns_query_name' contains as values nans or lists that only contain 1 string --> extract string from list
    train_df['dns_query_name'] = train_df['dns_query_name'].apply(extract_string)
    test_df['dns_query_name'] = test_df['dns_query_name'].apply(extract_string)

    #Creating My_Feature_Encoders
    dqn_mfe = My_Feature_Encoder(train_df, 'dns_query_name')
    hct_mfe = My_Feature_Encoder(train_df, 'http_content_type')
    hh_mfe = My_Feature_Encoder(train_df, 'http_host')

    #Preprocessing the Trainingsdataframe
    x_train, y_train, _, feature_order = preprocess(train_df, dqn_mfe, hct_mfe, hh_mfe, y_lab_enc)

    #Preprocessing the Testdataframe
    x_test, y_test, ids_test, _ = preprocess(test_df, dqn_mfe, hct_mfe, hh_mfe, y_lab_enc, feature_order)

    #Random Forest Fitting
    print("Random Forest Fitting")
    rfclf = RandomForestClassifier(n_estimators=100)
    rfclf.fit(x_train, y_train)

    #Making predictions based on x_test
    print("Making predictions based on x_test")
    predicted_results = rfclf.predict(x_test)
    
    #Saving results in json file
    create_json_results(y_lab_enc, predicted_results, ids_test, level)

    #Calculating and printing out accuracy
    accuracy = accuracy_score(y_test, predicted_results)
    print(f"🛈 Accuracy for {level}: {accuracy * 100:.2f}% 🛈")

    print("---------------------------------------------")