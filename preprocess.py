import pandas as pd
import ipaddress
import math

def transform_value(value, other_feature_value):
    if isinstance(value, list):
        if len(value) > 0:
            if isinstance(value[0], int):
                return sum(value) / other_feature_value
            else:
                total = 0

                for element in value:
                    total += int(element, 16)

                return total / other_feature_value
        else:
            return -1
    else:
        return -1
    
def split_lists(df, column): #Wegen absolut niedriger Feature Importance nach RandomForestFit | Verbraucht sehr viel Platz | Sehr viele Features
    '''
    max_len = max(df[column].apply(len))
    new_columns = [f"{column}_{i+1}" for i in range(max_len)]
    df[new_columns] = pd.DataFrame(df[column].tolist(), index=df.index)
    '''
    df = df.drop(column, axis=1)
    return df

def calculate_network_address(ip_addresses):
    if not ip_addresses or isinstance(ip_addresses, float) and math.isnan(ip_addresses):
        return -1
    ip_objects = []
    for ip in ip_addresses:
        try:
            ip_obj = ipaddress.IPv4Address(ip)
            ip_objects.append(ip_obj)
        except ipaddress.AddressValueError:
            pass
    if not ip_objects:
        return -1

    # Extrahiere die Binärrepräsentation der IP-Adressen und fülle fehlende Stellen mit 0 auf
    binary_ips = [format(int(ip), '032b') for ip in ip_objects]

    # Bestimme die gemeinsame Netzmaske im Binärformat
    common_mask_binary = ''.join('1' if all(bit == '1' for bit in bits) else '0' for bits in zip(*binary_ips))

    # Konvertiere die Binärform der Netzmaske direkt in eine Dezimalzahl (Integer)
    network_address_decimal = int(common_mask_binary, 2)

    return network_address_decimal

def preprocess(df, mfe1, mfe2, mfe3, y_lab_enc, feature_order = []):
    #Labels Encodieren
    df['label'] = y_lab_enc.transform(df['label'])

    #x_train und y_train trennen
    y = df['label']
    xdf = df.drop(columns=['label'])

    #x_train und ids trennen
    ids = xdf['id']
    xdf = xdf.drop(columns=['id'])
    
    #Irrelevante Features entfernen
    xdf = xdf.drop(columns=['dns_query_name_len', 'dns_answer_ttl', 'dns_query_class', 'dns_answer_cnt', 'dns_query_type', "dns_query_cnt", 'http_uri', 'da', 'sa'])

    ###NonObjects
    xdf = pd.get_dummies(xdf, columns=["tls_svr_cs_cnt", "tls_svr_key_exchange_len"])
    for feature, datentyp in xdf.dtypes.items():
        if datentyp != 'object':
            if xdf[feature].isna().any():
                xdf[feature] = xdf[feature].fillna(-1)

    ###Objects
    #TLS-Features
    xdf['tls_len_mean'] = xdf.apply(lambda row: transform_value(row['tls_len'], row['tls_cnt']), axis=1)
    xdf['tls_svr_len_mean'] = xdf.apply(lambda row: transform_value(row['tls_svr_len'], row['tls_svr_cnt']), axis=1)
    xdf['tls_cs_mean'] = xdf.apply(lambda row: transform_value(row['tls_cs'], row['tls_cs_cnt']), axis=1)
    xdf['tls_svr_cs_mean'] = xdf.apply(lambda row: transform_value(row['tls_svr_cs'], row['tls_svr_cs_cnt_1.0']), axis=1)
    xdf['tls_ext_types_mean'] = xdf.apply(lambda row: transform_value(row['tls_ext_types'], row['tls_ext_cnt']), axis=1)
    xdf['tls_svr_ext_types_mean'] = xdf.apply(lambda row: transform_value(row['tls_svr_ext_types'], row['tls_svr_ext_cnt']), axis=1)
    xdf = xdf.drop(columns=['tls_len', 'tls_svr_len', 'tls_cs', 'tls_svr_cs', 'tls_ext_types', 'tls_svr_ext_types'])

    #DNS- und HTTP-Features
    xdf['dns_query_name'] = xdf['dns_query_name'].apply(mfe1.enc)
    xdf['http_content_type'] = xdf['http_content_type'].apply(mfe2.enc)
    xdf['http_host'] = xdf['http_host'].apply(mfe3.enc)
    xdf['dns_answer_common_network_ip'] = xdf['dns_answer_ip'].apply(calculate_network_address)
    xdf = xdf.drop(columns=['dns_answer_ip'])

    #METADATA-Features
    for column in xdf.columns:
        if isinstance(xdf[column][0], list):
            if isinstance(xdf[column][0][0], int):
                xdf = split_lists(xdf, column)
    
    #x_train und x_test auf gleiche Feature-Reihenfolge bringen
    if len(feature_order) > 0:
        xdf = xdf.reindex(columns=feature_order)

    #Reihenfolge der Features
    xdf_columns = xdf.columns

    return xdf, y, ids, xdf_columns