import pandas as pd


def date_parser(date):
    return pd.to_datetime(date, format='%d.%m.%Y')


def load_data(data_path, parse_date=False):
    args = {}

    if parse_date:
        args = {
            'parse_dates': ['date'],
            'date_parser': lambda date: date_parser(date)
        }

    return pd.read_csv(data_path, **args)
