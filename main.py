import json
import pandas as pd
from my_feature_encoder import My_Feature_Encoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import time
from preprocess import preprocess

def extract_string(lst):
    return lst[0] if isinstance(lst, list) and len(lst) == 1 else lst

for level in ['toplevel_1', 'midlevel_2', 'finegrained_3']:
    #Trainingsdataframe erstellen
    with open('x_train.json', 'r') as file:
        x_train_points = [json.loads(line) for line in file]
    with open('y_train_' + level + '.json', 'r') as file:
        y_train_points = json.load(file)
    train_df = pd.DataFrame(x_train_points)
    train_df['label'] = train_df['id'].astype(str).map(y_train_points)

    #Testdataframe erstellen
    with open('x_test.json', 'r') as file:
        x_test_points = [json.loads(line) for line in file]
    with open('y_test_' + level + '.json', 'r') as file:
        y_test_points = json.load(file)
    test_df = pd.DataFrame(x_test_points)
    test_df['label'] = test_df['id'].astype(str).map(y_test_points)

    #Labelencoder erstellen
    y_lab_enc = LabelEncoder()
    y_lab_enc.fit(train_df['label'])

    #String aus Liste holen
    train_df['dns_query_name'] = train_df['dns_query_name'].apply(extract_string)
    test_df['dns_query_name'] = test_df['dns_query_name'].apply(extract_string)

    #Meine Feature Encoder erstellen
    mfe1 = My_Feature_Encoder(train_df, 'dns_query_name')
    mfe2 = My_Feature_Encoder(train_df, 'http_content_type')
    mfe3 = My_Feature_Encoder(train_df, 'http_host')

    #Trainingsdataframe preprocessen
    x_train, y_train, feature_order = preprocess(train_df, mfe1, mfe2, mfe3, y_lab_enc)

    #Random Forest fitten
    clf = RandomForestClassifier(n_estimators=1) #TODO 100
    print("Start")
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{elapsed_time}s gedauert")

    #Testdataframe preprocessen
    x_test, y_test, _ = preprocess(test_df, mfe1, mfe2, mfe3, y_lab_enc, feature_order)

    ##########################################################################
    predictionarr = clf.predict(x_test)

    def precision(xt, yt):
        a = 0
        b = len(yt)
        for i in range(len(yt)):
            if yt[i] == xt[i]:
                a += 1
        return a/b

    print("Precision: " + str(precision(predictionarr, y_test)))