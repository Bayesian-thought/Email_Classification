import os
import unittest
from ..dataset import *

user_folder_uri = os.path.join(os.getcwd(), 'email_classifier/data/beck-s')

class TestDataset(unittest.TestCase):
	def setUp(self):pass
		#self.data_initializer = DataInitializer(user_folder_uri)

	def test_get_emails(self):
		first_email = get_emails(user_folder_uri)[0]
		self.assertEqual(first_email.uri, os.path.join(user_folder_uri, 'analyst_assoc_program', '26.'))
		self.assertEqual(first_email.classification, 'analyst_assoc_program')
		self.assertEqual(first_email.datetime, 'Wed, 7 Jun 2000 07:17:00 -0700 (PDT)')
		self.assertEqual(first_email.header_from, 'Debbie Flores')
		self.assertEqual(first_email.header_to, 'Jeffrey A Shankman, Sally Beck')
		self.assertEqual(first_email.header_cc, 'Patti Thompson')
		self.assertEqual(first_email.header_bcc, '')
		self.assertEqual(first_email.subject, 'Associate/Analyst Information Lunch')
		self.assertEqual(first_email.get_words()[0], 'this')

	def test_get_word_counts(self):
		global_word_counts, class_word_counts = get_word_counts(self.data_initializer.emails)
		self.assertEqual(global_word_counts['enron'], 6655)
		self.assertEqual(class_word_counts['analyst_assoc_program']['enron'], 41)
		# TODO: remove, but for now it's nice to see what's in there
		print global_word_counts

	def test_get_name_counts(self):
		pass

	def test_create_feature_sets(self):
		pass

	def test_reduce_by_information_gain(self):
		emails = get_emails(user_folder_uri)
		word_features, name_features = create_feature_sets(emails)
		print word_features
		word_features, name_features = reduce_feature_sets(word_features, name_features, emails, 0.95, 'information gain')
		print word_features
		print name_features

	def test_reduce_by_chi_square(self):
		pass

	def test_example___str__(self):
		pass

	def test_data_initializer_preprocess(self):
		pass

	def test_data_initializer__parse_email(self):
		email = self.data_initializer.emails[0]
		example_type = self.data_initializer.get_example_type()
		example = self.data_initializer._parse_email(email, example_type)
		# TODO: remove
		print example
		for index, value in enumerate(example.input_vector):
			if value:
				print example.type.features[index] + ": " + str(value)

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestDataset))
    return suite

if __name__ == '__main__':
    unittest.main()
