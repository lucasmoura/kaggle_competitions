import unittest

from kaggleflow.manager.search_model import ModelSearcher, PipelineSearcher, MetricSearcher

from tests.test_models.linear_regression.model import Test
from tests.test_models.metric import TestMetric
from tests.test_models.linear_regression.pipelines.p1.pipeline import (LFillMissing,
                                                                       LTransformations,
                                                                       LCreate,
                                                                       LDrop,
                                                                       LFinalize)


class TestSearchModel(unittest.TestCase):

    def test_find_model(self):
        model_searcher = ModelSearcher('tests.test_models')
        model_name = 'linear_regression'
        model_path = model_searcher.find_module(model_name)
        self.assertEqual(model_path, 'tests.test_models.linear_regression')

        model_name = 'random_forest'
        model_path = model_searcher.find_module(model_name)
        self.assertEqual(model_path, 'tests.test_models.random_forest')

        model_name = 'neural_network'
        model_path = model_searcher.find_module(model_name)
        self.assertEqual(model_path, None)

    def test_find_pipeline(self):
        folder_to_look = 'tests.test_models.linear_regression.pipelines'
        pipeline_searcher = PipelineSearcher(folder_to_look)
        pipeline_name = 'p1'
        pipeline_path = pipeline_searcher.find_module(
            pipeline_name)
        self.assertEqual(
            pipeline_path,
            'tests.test_models.linear_regression.pipelines.p1'
        )

        pipeline_name = 'p5'
        pipeline_path = pipeline_searcher.find_module(
            pipeline_name)
        self.assertEqual(
            pipeline_path,
            None
        )

    def test_load_pipeline_operations(self):
        folder_to_look = 'tests.test_models.linear_regression.pipelines'
        pipeline_searcher = PipelineSearcher(folder_to_look)
        expected_order = [LFillMissing, LTransformations, LCreate, LDrop, LFinalize]

        operations = pipeline_searcher.get_class('p1')
        self.assertEqual(operations, expected_order)

    def test_load_model(self):
        model_searcher = ModelSearcher('tests.test_models')
        model = model_searcher.get_class('linear_regression')

        self.assertEqual(model, Test)

    def test_load_metric(self):
        metric_searcher = MetricSearcher('tests.test_models')
        metric_class = metric_searcher.get_class('metric')

        self.assertEqual(metric_class, TestMetric)
