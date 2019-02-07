import os
import unittest

from chemicalchecker.util import Config


class TestConfig(unittest.TestCase):

    def setUp(self):
        # path for test data
        test_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(test_dir, 'data')
        os.environ["CC_CONFIG"] = os.path.join(
            self.data_dir, 'config.json')

    def test_init(self):
        filename = os.path.join(self.data_dir, 'config.json')
        self.assertTrue(os.path.isfile(filename))
        config = Config(filename)
        self.assertTrue(hasattr(config, 'PATH'))
        self.assertTrue(hasattr(config.PATH, 'CC_ROOT'))
        self.assertEqual(config.PATH.CC_ROOT, '/aloy/web_checker/')

    def test_init_default(self):
        filename = os.path.join(self.data_dir, 'config.json')
        self.assertTrue(os.path.isfile(filename))
        config = Config(filename)
        self.assertTrue(hasattr(config, 'PATH'))
        self.assertTrue(hasattr(config.PATH, 'CC_ROOT'))
        self.assertEqual(config.PATH.CC_ROOT, '/aloy/web_checker/')

        config = Config()
        self.assertTrue(hasattr(config, 'PATH'))
        self.assertTrue(hasattr(config.PATH, 'CC_ROOT'))
        self.assertEqual(config.PATH.CC_ROOT, '/aloy/web_checker/')

        del os.environ['CC_CONFIG']
        with self.assertRaises(KeyError):
            config = Config()
        os.environ["CC_CONFIG"] = os.path.join(
            self.data_dir, 'config.json')
