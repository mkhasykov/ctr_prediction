import numpy as np
import os
import io
from dateutil.relativedelta import relativedelta
import requests
import json
import subprocess
import pickle

from utls import *


def get_reqs_ch(start_date, end_date, folder2wright):
    query = "select distinct * " + \
            "from (" + \
                "select req_ts, req_id, seq_id " + \
                "from kimberlite.rtb_subseq_log " + \
                f"where toDate(req_ts) >= '{start_date}' " + \
                f"and   toDate(req_ts) <= '{end_date}' " + \
                "and type = 'trchk'" + \
            ") es " + \
            "left join (" + \
                "with 1 as click " + \
                "select req_ts, req_id, seq_id, click " + \
                "from kimberlite.rtb_subseq_log " + \
                f"where toDate(req_ts) >= '{start_date}' " + \
                f"and   toDate(req_ts) <= '{end_date}' " + \
                "and type = 'click'" + \
            ") cs " + \
            "using req_ts, req_id, seq_id"
    df = try_redash(query)
    df.to_csv(os.path.join(folder2wright, 'reqs_ch.csv'), index=False)


def parse_ya_metrika(start_date, end_date, folder2wright):
    url    = "https://api-metrika.yandex.ru/stat/v1/data.csv"
    header = {'Authorization': 'OAuth ' + YA_OAUTH_TOKEN}
    rename_dict = {'Дата визита': 'date',
                   'UTM Content': 'bid_id', 
                   'Визиты': 'visits', 
                   'Просмотры': 'pageviews',
                   'Целевые визиты (Клик по ссылке)': 'clicks'}

    days = days_between(start_date, end_date, '%Y-%m-%d')
    transit_df = pd.DataFrame()
    for day in days:
        params = {
            'metrics':    'ym:s:goal46984879visits',
            'dimensions': 'ym:s:date,ym:s:UTMContent',
            'filters':    'ym:s:UTMTerm!.(7334160, 7934232, 7334164)',
            'date1':      day,
            'date2':      day,
            'limit':      100000,
            'ids':        YA_COUNTER_ID
        }
        
        s  = requests.get(url, params=params, headers=header).content # Получаем данные по URL
        df = pd.read_csv(io.StringIO(s.decode('utf-8'))) # Записываем данные в датафрейм
        df = df.rename(rename_dict, axis = 'columns')
        df = df.iloc[1:]
        df = df.loc[df['bid_id']!='Не определено']
        transit_df = transit_df.append(df, ignore_index=True)
        del df
    
    transit_df.to_csv(os.path.join(folder2wright, 'metrika_events.csv'), index=False)


def parse_amplitude(start_date, end_date, folder2wright):
    # константные части api-запроса
    events  = ['ClickRowTeaser', 'ClickTextTeaser']
    filters = [['site_id', 'is not', [7334160, 7934232, 7334164]]]
    groups  = ['bid_id', 'variant_id']

    row_teaser_df = pd.DataFrame()
    for date in days_between(start_date, end_date, "%Y%m%d"):
        for event in events:
            segmentation = segment_builder(event, filters, groups)
            request = amplitude_request(date, segmentation)
            # отправка запроса и парсинг полученных данных
            o = subprocess.check_output(request, shell=True).decode()
            j = json.loads(o)
    
            num_rows = len(j['data']['series'])
            data = {'date':       j['data']['xValues'] * num_rows,
                    'event':      [event] * num_rows}
            for i, feature in enumerate(groups):
                data[feature] = [k[1].split('; ')[i] for k in j['data']['seriesLabels']]
            row_teaser_df = row_teaser_df.append(pd.DataFrame.from_dict(data), ignore_index=True)

    row_teaser_df.to_csv(os.path.join(folder2wright, 'amplitude_events.csv'), index=False)
    

def get_reqs_metrika_ampl(folder):
    bids_list = []
    for file in ['metrika_events.csv', 'amplitude_events.csv']:
        bids_list += pd.read_csv(os.path.join(folder, file), usecols=['bid_id'], squeeze=True).tolist()
    bids_list = list(set(bids_list))
    
    df = pd.DataFrame()
    for i, batch in enumerate(list_split(bids_list)):
        query = "select req_ts, req_id, seq_id from kimberlite.rtb_impbid " + \
                "where uid in (" + \
                ",".join(list(map(lambda x: "'{}'".format(x), batch))) + \
                ")"
        df = df.append(try_redash(query), ignore_index=True, sort=False)
    
    df.to_csv(os.path.join(folder, 'reqs_metrika_ampl.csv'), index=False)
    
    
def unite_reqs(folder):    
    df = pd.read_csv(os.path.join(folder, 'reqs_metrika_ampl.csv'))
    df = df.append(pd.read_csv(os.path.join(folder, 'reqs_ch.csv')), ignore_index=True, sort=False)
    df = df.fillna(0)
    df = df.groupby(['req_ts', 'req_id', 'seq_id']).agg('max').reset_index()
    df['click'] = df['click'].astype(int)
    df.to_csv(os.path.join(folder, 'reqs_merged.csv'), index=False)


def parse_raw_features(folder):
    with open(os.path.join(folder, "reqs_merged.csv")) as f:
        reqs_list = f.readlines()
    reqs_list = ['_'.join(line.strip().split(',')[:-1])for line in reqs_list][1:]
    
    # parse features from rtb_impbid
    df_impbid = pd.DataFrame()
    for i, batch in enumerate(list_split(reqs_list, size=5000)):
        query = "select req_ts, req_id, seq_id, format_id, variant_id " + \
                "from kimberlite.rtb_impbid " + \
                "where concat(toString(req_ts), '_', toString(req_id), '_', toString(seq_id)) in (" + \
                        ",".join(list(map(lambda x: "'{}'".format(x), batch))) + \
                ")"
        df_impbid = df_impbid.append(try_redash(query), ignore_index=True)
    df_impbid = df_impbid.drop_duplicates()
    
    # parse features from rtb_impbid
    reqs_list = [line.rsplit('_', 1)[0] for line in reqs_list]
    df_req = pd.DataFrame()
    for i, batch in enumerate(list_split(reqs_list, size=5000)):
        query = "select ts, id, country_id, device_id, site_id, ua " + \
                "from kimberlite.rtb_req " + \
                "where concat(toString(ts), '_', toString(id)) in (" + \
                        ",".join(list(map(lambda x: "'{}'".format(x), batch))) + \
                ")"
        df_req = df_req.append(try_redash(query), ignore_index=True)
    df_req = df_req.drop_duplicates()
    
    # merge features
    df = pd.merge(df_req, df_impbid, left_on=['ts', 'id'], right_on=['req_ts', 'req_id'])
    df.to_csv(os.path.join(folder, 'raw_features.csv'), index=False)


def parse_blacklist(folder2wright):
    query = "select distinct id from ssp_site" + \
            " where is_black='true'"
    sites_blocked = try_redash(query, data_source='pg')
    sites_blocked = set(sites_blocked.squeeze())
    with open(os.path.join(folder2wright, 'sites_blocked.txt'), "wb") as fp:
        pickle.dump(sites_blocked, fp)
    

def prepare_train_dataset(folder):
    df = pd.read_csv(os.path.join(folder, 'raw_features.csv'))
    df = df[~df['site_id'].isin([7334160, 7934232, 7334164])]
    
    df = user_agent_parse(df)
    df = df.rename(index=str, columns={"user_agent.family": "browser", 
                                       "device.family":     "device_family", 
                                       "os.family":         "os"})
    features  = ['country_id', 'device_id', 'format_id', 'browser', 'site_id', 'variant_id', 'device_family', 'os']
    df = df[['req_ts', 'req_id', 'seq_id'] + features]
    
    # add target_values
    targets_df = pd.read_csv(os.path.join(folder, 'reqs_merged.csv'))
    df = pd.merge(df, targets_df, on=['req_ts', 'req_id', 'seq_id'])
    del targets_df
    
    # merge with old data
    if os.path.isfile(os.path.join(folder, 'train.csv')):
        df_old = pd.read_csv(os.path.join(folder, 'train.csv'))
        df = pd.concat([df, df_old], ignore_index=True, sort=False)
        df = df.drop_duplicates(subset=['req_ts', 'req_id', 'seq_id'])
        del old_df
    
    # drop old observations (elder than 4 weeks)
    df['req_ts'] = pd.to_datetime(df['req_ts'])
    df = df.loc[df['req_ts'] >= (df['req_ts'].max() - relativedelta(weeks=4))]
    df = df.sort_values('req_ts', ascending=False)
    
    # delete blacklists sites
    with open(os.path.join(folder, 'sites_blocked.txt'), "rb") as fp:
        sites_blocked = pickle.load(fp)
    df = df.loc[~df['site_id'].isin(sites_blocked)]
    
    # save prepared train dataset
    df.to_csv(os.path.join(folder, 'train.csv'), index=False)
    
