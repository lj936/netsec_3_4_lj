import ipaddress
import math

def calculate_mean(value, other_feature_value):
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

    binary_ips = [format(int(ip), '032b') for ip in ip_objects]
    common_mask_binary = ''.join('1' if all(bit == '1' for bit in bits) else '0' for bits in zip(*binary_ips))
    network_address_decimal = int(common_mask_binary, 2)

    return network_address_decimal

def preprocess(df, dqn_mfe, hct_mfe, hh_mfe, y_lab_enc, feature_order = []):
    #Encoding Labels
    df['label'] = y_lab_enc.transform(df['label'])

    #Separating x, y and id's
    y = df['label']
    ids = df['id']
    xdf = df.drop(columns=['label', 'id'])
    
    #Dropping irrelevant features
    xdf = xdf.drop(columns=['dns_query_name_len', 'dns_answer_ttl', 'dns_query_class', 'dns_answer_cnt', 'dns_query_type', "dns_query_cnt", 'http_uri', 'da', 'sa', "rev_intervals_ccnt", "pld_ccnt", "rev_hdr_ccnt", "rev_pld_ccnt", "rev_ack_psh_rst_syn_fin_cnt", "ack_psh_rst_syn_fin_cnt", "intervals_ccnt", "hdr_ccnt"])

    ###Preprocessing Numeric Objects
    #Filling all missing values with -1
    for feature, datentyp in xdf.dtypes.items():
        if datentyp != 'object':
            if xdf[feature].isna().any():
                xdf[feature] = xdf[feature].fillna(-1)

    ###Preprocessing Non-Numeric Objects
    #Calculating means out of TLS-Array-Features
    xdf['tls_len_mean'] = xdf.apply(lambda row: calculate_mean(row['tls_len'], row['tls_cnt']), axis=1)
    xdf['tls_svr_len_mean'] = xdf.apply(lambda row: calculate_mean(row['tls_svr_len'], row['tls_svr_cnt']), axis=1)
    xdf['tls_cs_mean'] = xdf.apply(lambda row: calculate_mean(row['tls_cs'], row['tls_cs_cnt']), axis=1)
    xdf['tls_svr_cs_mean'] = xdf.apply(lambda row: calculate_mean(row['tls_svr_cs'], row['tls_svr_cs_cnt']), axis=1)
    xdf['tls_ext_types_mean'] = xdf.apply(lambda row: calculate_mean(row['tls_ext_types'], row['tls_ext_cnt']), axis=1)
    xdf['tls_svr_ext_types_mean'] = xdf.apply(lambda row: calculate_mean(row['tls_svr_ext_types'], row['tls_svr_ext_cnt']), axis=1)
    xdf = xdf.drop(columns=['tls_len', 'tls_svr_len', 'tls_cs', 'tls_svr_cs', 'tls_ext_types', 'tls_svr_ext_types'])

    #Feature Encoding DNS- and HTTP-String-Features
    xdf['dns_query_name'] = xdf['dns_query_name'].apply(dqn_mfe.enc)
    xdf['http_content_type'] = xdf['http_content_type'].apply(hct_mfe.enc)
    xdf['http_host'] = xdf['http_host'].apply(hh_mfe.enc)
    xdf['dns_answer_common_network_ip'] = xdf['dns_answer_ip'].apply(calculate_network_address)
    xdf = xdf.drop(columns=['dns_answer_ip'])
    
    ###Feature Order
    #Bring x_train and x_test to the same feature order
    if len(feature_order) > 0:
        xdf = xdf.reindex(columns=feature_order)

    #feature order
    xdf_columns = xdf.columns

    return xdf, y, ids, xdf_columns