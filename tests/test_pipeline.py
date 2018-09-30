import unittest

from kaggleflow.preprocessing.pipeline import Operation


class TestPipeline(unittest.TestCase):

    def test_pipeline_public_methods(self):

        class Test(Operation):

            def test1(self):
                pass

            def test2(self):
                pass

        test_class = Test(signature='test')
        signature_methods = test_class.get_signature_methods()
        expected_signature_methods = ['test1', 'test2']

        self.assertEqual(signature_methods, expected_signature_methods)

        class Test2(Operation):
            pass

        test_class = Test2(signature='test')
        signature_methods = test_class.get_signature_methods()
        expected_signature_methods = []

        self.assertEqual(signature_methods, expected_signature_methods)

    def test_run_methods(self):

        class Test(Operation):

            def test1(self):
                self.data[0].append(1)

            def test2(self):
                self.data[0].append(2)

        data = []
        test_class = Test(signature='test')
        test_class.set_dataset(data, None, None)
        test_class.run()

        self.assertEqual(data, [1, 2])
