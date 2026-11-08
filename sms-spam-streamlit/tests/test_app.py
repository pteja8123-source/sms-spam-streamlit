"""Exercise the actual Streamlit UI without opening a browser."""

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


class AppTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(APP), default_timeout=30).run()
        self.assertEqual(len(self.app.exception), 0)

    def submit(self, text):
        self.app.text_area[0].set_value(text)
        self.app.button[0].click().run()
        self.assertEqual(len(self.app.exception), 0)

    def test_blank_message(self):
        self.submit("   ")
        self.assertEqual(self.app.warning[0].value, "Enter a message before checking.")
        self.assertEqual(len(self.app.metric), 0)

    def test_ham_message(self):
        self.submit("Hey, are we still meeting for lunch tomorrow?")
        self.assertEqual(self.app.success[0].value, "Prediction: Legitimate message (ham)")
        self.assertEqual(self.app.metric[0].label, "Model spam score")

    def test_spam_message(self):
        self.submit("Congratulations! You have won a FREE cash prize. Call now to claim your reward!")
        self.assertEqual(self.app.warning[0].value, "Prediction: Spam")

    def test_unknown_terms(self):
        self.submit("qzxvptnm jxqzvtnm")
        self.assertIn("no terms recognized", self.app.warning[0].value)
        self.assertEqual(len(self.app.metric), 0)

    def test_example_fills_input(self):
        self.app.selectbox[0].select("Prize message").run()
        self.assertIn("FREE cash prize", self.app.text_area[0].value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
