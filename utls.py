import os
import datetime
import time
from dateutil.rrule import rrule, MONTHLY
from dateutil.relativedelta import relativedelta
import calendar
from copy import copy

import pandas as pd
from ua_parser import user_agent_parser
from redashAPI.client import RedashAPIClient


# Auth Keys for YaMetrika
YA_OAUTH_TOKEN = ''
YA_COUNTER_ID  = ''

# Auth Keys for Amplitude
AMPL_API_KEY    = ''
AMPL_SECRET_KEY = ''

# Auth Keys for Redash
REDASH_API_KEY  = ''
REDASH_HOST     = ''


def check_present_data(folder):
    '''
    Determine date range of data to be loaded
    '''
    now = datetime.datetime.now()
    end_date = now.strftime("%Y-%m-%d")
    if os.path.isdir(folder):
        if os.path.isfile(os.path.join(folder, 'bid_df_prepared.csv')):
            # last observation in present data
            last_ts = pd.read_csv(os.path.join(folder, 'bid_df_prepared.csv'), nrows=1).at[0, 'ts']
            last_ts = datetime.datetime.strptime(last_ts, '%Y-%m-%d %H:%M:%S')
            if last_ts > (now - relativedelta(weeks=4)):
                start_date = (last_ts - relativedelta(days=1)).strftime("%Y-%m-%d")
                return start_date, end_date
    else:
        os.makedirs(folder)
    start_date = (now - relativedelta(weeks=4)).strftime("%Y-%m-%d")
    return start_date, end_date


def periods_between(start_date, end_date):
    strt_dt = datetime.date(int(start_date[:4]), int(start_date[5:7]), int(start_date[8:10]))
    end_dt  = datetime.date(int(end_date[:4]), int(end_date[5:7]), int(end_date[8:10]))
    
    months = [dt.strftime("%Y-%m") for dt in rrule(MONTHLY, dtstart=strt_dt, 
                                                   until=end_dt)]
    lst_days = [calendar.monthrange(int(month[:4]), int(month[5:7]))[1] for month in months]
    periods = [[month+'-01', month+'-'+str(lst_days[i])] for i, month in enumerate(months)]
    
    periods[0][0]   = copy(start_date)
    periods[-1][-1] = copy(end_date)
    return periods


def days_between(start_date, end_date, str_format):
    d1 = datetime.date(int(start_date[:4]), int(start_date[5:7]), int(start_date[8:10]))
    d2 = datetime.date(int(end_date[:4]), int(end_date[5:7]), int(end_date[8:10]))
    
    delta = d2 - d1
    return [(d1 + datetime.timedelta(days=i)).strftime(str_format) for i in range(delta.days + 1)]


def try_redash(query, num_attempts=20, data_source='ch'):
    ''' Parse query result from Redash

        INPUT:
            query: text of query
            num_attempts: number of consecutive attempts 
                          to make in case of parsing failure
            data_sourse: ch - 'ClickHouse', 'pg' - PostgreSQL
    '''
    if data_source not in ['ch', 'pg']:
        raise Exception("Data source should be 'ch' for ClickHouse of 'pg' for PostgreSQL.")
    
    ds_ids = {'pg': 1, 
              'ch': 8}
    
    Redash = RedashAPIClient(REDASH_API_KEY, REDASH_HOST)
    
    attempts = 0
    while attempts < num_attempts:
        try:
            res = Redash.generate_query_result(ds_ids[data_source], query, 1)
            s   = res.json()['query_result']['data']
            break
        except KeyError:
            attempts += 1
            time.sleep(10)
            continue
    if attempts>=num_attempts:
        raise Exception("Current Redash API-query was failed {} consecutive times".format(num_attempts))
    return pd.DataFrame.from_dict(s['rows'])


def segment_builder(event, filters, groups):
    '''
    Формирование сегмента в запросе для Amplitude
    '''
    # определение ивента
    request = 'e=\{\\"event_type\\":\\"' + event + '\\"'
    # определение фильтров
    if filters:
        request += ',\\"filters\\":\['
        filtr_components = []
        for filtr in filters:
            filtr_components += ['\{\\"subprop_type\\":\\"event\\",' + \
                                   '\\"subprop_key\\":\\"' + filtr[0] + '\\",' + \
                                   '\\"subprop_op\\":\\"' + filtr[1].replace(' ', '+') + '\\",' + \
                                   '\\"subprop_value\\":\[' + ','.join(['\\"{}\\"'.format(x) for x in filtr[2]]) + '\]\}']
        request += ','.join(filtr_components)
        request += '\]'
    # определение группировки
    if groups:
        request += ',\\"group_by\\":\['
        group_components = []
        for group in groups:
            group_components += ['\{\\"type\\":\\"event\\",' + \
                                   '\\"value\\":\\"' + group + '\\"\}']
        request += ','.join(group_components)
        request += '\]'
    request += '\}'
    return request

def amplitude_request(date, segmentation,
                      AMPL_API_KEY = '', 
                      AMPL_SECRET_KEY = ''):
    start = "curl -s -u "
    url   = "https://amplitude.com/api/2/events/segmentation?"
    left = "&m=totals&start=" + date + "&end=" + date
    
    request = (start + 
               AMPL_API_KEY + ':' + AMPL_SECRET_KEY + ' "' + 
               url +
               segmentation + 
               left + '"')
    
    return request
 
def list_split(list_init, size=15000):
    '''
    Split list of elems into list of disjoint lists
    Each inner list contains not more than "size" unique elems
    '''
    for i in range(len(list_init) // size):
        yield list_init[i*size:(i+1)*size]
    if (len(list_init) % size) > 0:
        yield list_init[size*(len(list_init) // size):len(list_init)]
    

def user_agent_parse(data):
    data_inf = data.loc[~data['ua'].isnull(), 'ua'].copy()
    data_inf = data_inf.loc[~data_inf.duplicated()]
    data_inf = pd.io.json.json_normalize(data_inf.map(user_agent_parser.Parse))
    data_inf = data_inf.rename({'string': 'ua'}, axis = 'columns')
    
    data = pd.merge(data, data_inf, on='ua', how='left')
    return data

