import argparse

import pandas as pd


def create_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-icp',
                        '--item-categories-path',
                        type=str)

    parser.add_argument('-ip',
                        '--items-path',
                        type=str)

    parser.add_argument('-stp',
                        '--sales-train-path',
                        type=str)

    parser.add_argument('-sp',
                        '--shops-path',
                        type=str)

    parser.add_argument('-tp',
                        '--test-path',
                        type=str)

    return parser


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


def main():
    parser = create_argparser()
    user_args = vars(parser.parse_args())

    sales_train = load_data(
        user_args['sales_train_path'], parse_date=True)
    item_categories = load_data(
        user_args['item_categories_path'])
    items = load_data(
        user_args['items_path'])
    shops = load_data(
        user_args['shops_path'])
    test = load_data(
        user_args['test_path'])

    """
    data info:

        date:           date in format dd/mm/yyyy
        date_block_num: a consecutive month number, used for convenience.
                        January 2013 is 0, February 2013 is 1,...,
                        October 2015 is 33
        shop_id:        unique identifier of a shop
        item_id:        unique identifier of a product
        item_price:     current price of an item
        item_cnt_day:   number of products sold. You are predicting a
                        monthly amount of this measure
    """
    print(sales_train.head())
    print()

    """
    data info:

       item_category_name: name of item category
       item_category_id:   unique identifier of item category
    """
    print(item_categories.head())
    print()

    """
    data info:

       item_name: name of item
       item_id: unique identifier of a product
       item_category_id: unique identifier of item category
    """
    print(items.head())
    print()

    """
    data info:

        shop_name: name of shop
        shop_id:   unique identifier of a shop
    """
    print(shops.head())
    print()

    """
    data info:

        ID:        an Id that represents a (Shop, Item)
                   tuple within the test set
        shop_id:   unique identifier of a shop
        item_id:   unique identifier of a product
    """
    print(test.head())
    print()


if __name__ == '__main__':
    main()
