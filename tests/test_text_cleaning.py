import unittest
from nlp_data_cleaning_and_transformation_project import clean_text_basic, clean_text_advanced

class TestTextCleaning(unittest.TestCase):
    def test_basic_cleaning(self):
        text = "<b>Hello!</b> This is a test, number 123."
        cleaned = clean_text_basic(text)
        self.assertTrue("hello" in cleaned)
        self.assertTrue("test" in cleaned)
        self.assertFalse("<b>" in cleaned)
        self.assertFalse("123" in cleaned)

    def test_advanced_cleaning(self):
        text = "The cats are sitting on the mat."
        cleaned = clean_text_advanced(text)
        self.assertFalse("the" in cleaned)  # stopword removed
        self.assertIn("cat", cleaned)       # lemmatized

if __name__ == "__main__":
    unittest.main()