import pandas as pd

from utils.path import create_path


def generate_submission(predictions, id_column,
                        target_column, id_values, save_path):
    submission_df = pd.DataFrame(
        {id_column: id_values,
         target_column: predictions}
    )

    submission_df.to_csv(
        create_path(save_path, 'submission.csv'),
        index=False
    )
