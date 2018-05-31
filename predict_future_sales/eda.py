import argparse

from dataset.dataset import load_data


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


def print_data(sales_train, item_categories, items, shops, test):
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
                        monthly amount of this measure (*Sometimes this
                        variables are negative, that may indicate
                        refunds on that day, maybe I should remove negative
                        variables from the train data ?)
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


def print_data_info(sales_train, item_categories, items, shops, test):
    print('Size of datasets:\n')
    print(
        'train: {}\ntest: {}\nitems cat: {}\nitems: {}\nshops: {}\n'.format(
            sales_train.shape, test.shape, item_categories.shape,
            items.shape, shops.shape)
    )


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

    print_data(sales_train, item_categories, items, shops, test)
    print_data_info(sales_train, item_categories, items, shops, test)


if __name__ == '__main__':
    main()
