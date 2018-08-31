import unittest

from manager.search_model import find_model, find_pipeline


class TestSearchModel(unittest.TestCase):

    def test_find_model(self):
        model_name = 'linear_regression'
        model_path = find_model(model_name, module='tests/test_models')
        self.assertEqual(model_path, 'tests.test_models.linear_regression')

        model_name = 'random_forest'
        model_path = find_model(model_name, module='tests/test_models')
        self.assertEqual(model_path, 'tests.test_models.random_forest')

        model_name = 'neural_network'
        model_path = find_model(model_name, module='tests/test_models')
        self.assertEqual(model_path, None)

    def test_find_pipeline(self):
        model_path = 'tests.test_models.linear_regression'
        pipeline_name = 'p1'
        pipeline_path = find_pipeline(pipeline_name, model_path)
        self.assertEqual(
            pipeline_path,
            'tests.test_models.linear_regression.pipelines.p1'
        )

        model_path = 'tests.test_models.linear_regression'
        pipeline_name = 'p5'
        pipeline_path = find_pipeline(pipeline_name, model_path)
        self.assertEqual(
            pipeline_path,
            None
        )
