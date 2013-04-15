class Classifier(object):
	def __init__(self):
		self.classifications = []
		self.features = []
		self.example_type = None
		self.examples = []
		self.test_set = []

	def load_examples_into_set(self, file_uri, example_set):
		with open(file_uri, 'rb') as example_file:
			data = csv.reader(example_file)

			for example_row in data:
				id = None
				classification = None
				input_vector = []
				for col_index, col in enumerate(example_row):
					if col_index == 0:
						id = col
					elif col_index == 1:
						classification = col
					else:
						input_vector.append(col)

				example = self.example_type.create_example(id, input_vector, classification)
				example_set.append(example)

	def load_training_examples(self, training_file_uri):
		self.examples = []
		self.load_examples_into_set(training_file_uri, self.examples)

	def load_test_set(self, test_file_uri):
		self.test_set = []
		self.load_examples_into_set(test_file_uri, self.test_set)

	def train(self, training_file_uri=None):
		if training_file_uri:
			self.load_training_examples(training_file_uri)

		self.learn(self.examples, self.features)

	def test(self, test_file_uri=None):
		if test_file_uri:
			self.load_test_set(test_file_uri)

		total = len(self.test_set)
		correct = 0
		for test in self.test_set:
			if self.classify(test) == test.output:
				correct += 1

		return (correct, total)

	def classify(self, example):
		raise NotImplementedError("Override me for your learning algorithm!")

	def learn(self, examples, features):
		raise NotImplementedError("Override me for your learning algorithm!")