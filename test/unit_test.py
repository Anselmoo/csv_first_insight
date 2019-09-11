import unittest
import test

class TestMethods(unittest.TestCase):

    def test_case_data(self):
        #print(test.dataread.data_corl())
        el = ['a','b','c']
        self.assertEqual(len(test.sklsetups.elements(el)), len(el))

if __name__ == '__main__':
    unittest.main()