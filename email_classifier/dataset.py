import re
import os
import math
import sys
from email import Email

class DataInitializer(object):
    """
        Class that will be used to set up the feature vector on a per-user basis
        
        Arguments (for constructor):
            user_folder_uri       :-  Path to folder representing a user in the dataset
            reduce_using          :-  Strategy for reducing feature sets

        Goal:
            Return a matrix of of the form |class|feature1|feature2|...|
                                           |...  | ...    | ...    |...|

            It now returns a list of examples, which are themselves representations of
            the above form. So, a matrix + the overhead of many, many objects.

    """
    def __init__(self, user_folder_uri, reduce_by, reduce_using='information gain'):
        self.user_folder_uri = user_folder_uri
        # load the emails in the user's folder
        self.emails = get_emails(user_folder_uri)
        self.total_num_emails = len(self.emails)
        # store the emails per classification
        self.classification_emails = {}
        for classification in os.listdir(self.user_folder_uri):
            self.classification_emails[classification] = filter(lambda e: e.classification == classification, self.emails)
        # pull out the (term) features we are interested in from the dataset
        word_features = extract_feature_set(self.emails, reduce_by, reduce_using)
        self.word_features = word_features
        self.build_doc_counts_per_word()

    def build_doc_counts_per_word(self):
        self.doc_counts_per_feature = {}
        for index, feature in enumerate(self.word_features):
            print "Computing doc counts for feature (%d/%d): %s" % (index, len(self.word_features), feature)
            num_docs_with_feature = 0
            for email in self.emails:
                if email.contains(feature):
                    num_docs_with_feature += 1
            self.doc_counts_per_feature[feature] = num_docs_with_feature

    def get_example_type(self):
        classifications = os.listdir(self.user_folder_uri)
        # give 'special' features __ pre and postfixes to avoid conflict
        # with word features
        features = ['__month__', '__time-of-day__']
        for word_feature in self.word_features:
            features.append(word_feature)
        return ExampleType(classifications, features)

    def preprocess(self, feature_measure_method="tfidf"):
        """
            Constructs an Example (input vector + output) for each email and returns:
                ExampleType, (List of) Examples
        """
        # build up the metadata for each example
        example_type = self.get_example_type()

        # for each email, create an example
        examples = []
        for index, email in enumerate(self.emails):
            print "Converting email %d into example" % index
            examples.append(self._parse_email(email, example_type, feature_measure_method))

        return example_type, examples

    def _parse_email(self, email, example_type, feature_measure_method):
        """
            Returns an Example representing the given email
        """
        # FIXME: not a fan of this method or its hard-coded assumptions

        input_vector = []
        input_vector.append(email.get_month())
        input_vector.append(email.get_time_of_day())
        for index in range(2, len(example_type.features)):
            feature = example_type.features[index]
            value = 0
            if feature_measure_method == "tfidf":
                value = self._tf_idf(feature, email)
            elif feature_measure_method == "tf":
                value = self._naive_counter(feature, email)
            elif feature_measure_method == "boolean":
                value = self._boolean_counter(feature, email)
            input_vector.append(value)

        return example_type.create_example(email.uri, input_vector, email.classification)

    def _naive_counter(self, word, document):
        return document.count(word)

    def _boolean_counter(self, word, document):
        if document.contains(word):
            return 1
        return 0

    def _tf_idf(self, word, document):
        """
            Computes the tf-idf value for a
            'word' in training example 'document'
        """
        tf = self._term_freq(word, document)
        idf = self._inverse_doc_freq(word)
        return tf * idf 

    def _term_freq(self, word, document):
        """
            (number_of_times_word_appears_in_document)
            divided by
            (max_number_of_times_any_word_appears_in_document)
        """
        return document.count(word) / (1.0 * document.get_length())

    def _inverse_doc_freq(self, word):
        """
                log( num_docs / (1 + num_docs_with_word) )
        """
        num_docs_with_word = self.doc_counts_per_feature[word]
        idf = math.log(self.total_num_emails / (1 + 1.0 * num_docs_with_word))
        return idf

class ExampleType(object):
    """
        A factory for generating examples that all fit a certain profile, i.e. have
        the same input vectors and possible classifications.

        Essentially, it is a class for holding metadata about examples.
    """
    def __init__(self, classifications, features):
        self.classifications = classifications
        self.features = features
        self.feature_lookup = {}
        for index, feature in enumerate(self.features):
            self.feature_lookup[feature] = index

    def get_index(self, feature):
        return self.feature_lookup[feature]

    def create_example(self, id, input_vector, output):
        return Example(self, id, input_vector, output)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str([feature.name for feature in self.features])

class Example(object):
    """
        An instance from a dataset, associated with an ExampleType
    """
    def __init__(self, type, id, input_vector, output):
        self.type = type
        self.id = id
        self.input_vector = input_vector
        self.output = output

    def get_value(self, feature):
        feature_index = self.type.get_index(feature)
        if 0 <= feature_index < len(self.input_vector):
            return self.input_vector[feature_index]
        return None

def extract_feature_set(emails, reduce_by, using='information gain'):
    """
        Given a list of emails, extracts word features from them

        The feature set is reduced using the strategy provided as the
        'using' argument.
    """
    word_features = create_feature_set(emails)
    return reduce_feature_sets(word_features, emails, reduce_by, using)

def create_feature_set(emails):
    """ 
        Given a list of emails, returns two sets of features: words and names
    """
    global_word_count = get_word_counts(emails)[0]

    word_features = set()

    for word in global_word_count.keys():
        word_features.add(word)

    # make the set immutable now
    return frozenset(word_features)

def reduce_feature_sets(word_features, emails, amount, using='information gain'):
    """
        Uses information gain / chi-square to narrow down the word and name features
    """
    global_word_count, class_word_count = get_word_counts(emails)

    classifications = set(class_word_count.keys())

    if using == 'information gain':
        return reduce_by_information_gain(word_features, classifications,
                                          global_word_count, class_word_count, 
                                          amount)
    elif using == 'chi square':
        return reduce_by_chi_square(word_features, classifications,
                                    global_word_count, class_word_count, 
                                    amount)
    elif using == 'document frequency':
        return reduce_by_document_frequency(word_features, classifications,
                                            global_word_count, class_word_count,
                                            amount)
    return word_features

def reduce_by_percent(feature_list, amount):
    new_word_features = set()

    # keep only the top 1-amount percent
    k_word = int(len(feature_list) * (1 - amount))
    for i in range(k_word):
        new_word_features.add(feature_list[i][1])

    # make the new set immutable
    return frozenset(new_word_features)

def reduce_by_information_gain(word_features, classifications, 
                               global_word_count, class_word_count, 
                               amount):
    i = 0
    word_feature_info_gains = []
    for word_feature in word_features:
        word_feature_info_gain = information_gain(word_feature, classifications, 
                                                  global_word_count, class_word_count)
        word_feature_info_gains.append((word_feature_info_gain, word_feature))

        print "word (%d/%d): %s, %f" % (i, len(word_features), word_feature, word_feature_info_gain)

        i += 1

    # we want the highest information gain values first
    word_feature_info_gains.sort(reverse=True)

    return reduce_by_percent(word_feature_info_gains, amount)


def reduce_by_chi_square(word_features, classifications, 
                         global_word_count, class_word_count, 
                         amount):
    new_word_features = set(word_features)

    # TODO: chi square

    # make the new set immutable
    return frozenset(new_word_features)

def reduce_by_document_frequency(word_features, classifications,
                                global_word_count, class_word_count,
                                amount):
    i = 0
    word_feature_doc_freqs = []
    for word_feature in word_features:
        word_feature_doc_freq = term_probability(word_feature, global_word_count)
        word_feature_doc_freqs.append((word_feature_doc_freq, word_feature))

        print "word (%d/%d): %s, %f" % (i, len(word_features), word_feature, word_feature_doc_freq)

        i += 1

    # we want the highest probability values first
    word_feature_doc_freqs.sort(reverse=True)
    
    return reduce_by_percent(word_feature_doc_freqs, amount)

def get_emails(user_folder_uri):
    """
        Returns a list of all the emails inside the given user folder
    """
    emails = []
    for classification_folder in os.listdir(user_folder_uri):
        for root, dirs, files in os.walk(os.path.join(user_folder_uri, 
                                                      classification_folder)):
            for f in files:
                email_uri = os.path.join(user_folder_uri, classification_folder, f)
                email = Email(email_uri)
                emails.append(email)
    return emails

def get_word_counts(emails):
    """
        Looks at all emails and creates two dictionaries: one containing
        the count of all words per classification, and the other containing 
        the count of all words over the entire dataset
    """
    global_word_count = {}
    class_word_count = {}

    # loop through all the emails and build up the global dictionary
    for index, email in enumerate(emails):
        # add a dictionary for each class
        if email.classification not in class_word_count:
            class_word_count[email.classification] = {}

        print "Getting words from email %d" % index
        for word in email.get_words():
            if not word:
                continue

            if word not in global_word_count:
                global_word_count[word] = 1
            else:
                global_word_count[word] += 1

    # now that we have the global dictionary, initialize
    # each term's count in each classification to 0
    for classification in class_word_count:
        for word in global_word_count:
            class_word_count[classification][word] = 0

    # loop through all the emails again and build up the class dictionaries
    for email in emails:
        for word in email.get_words():
            if not word:
                continue

            class_word_count[email.classification][word] += 1

    return global_word_count, class_word_count

def information_gain(term, classifications, global_term_count, class_term_count):
    class_entropy = 0.0
    for classification in classifications:
        prob_class = class_probability(classification, global_term_count, class_term_count)
        if prob_class > 0:
            class_entropy += prob_class * math.log(prob_class)
    class_entropy = -class_entropy

    term_entropy = 0.0
    for classification in classifications:
        prob_class_given_term = class_probability_given_term(classification, term, global_term_count, class_term_count)
        if prob_class_given_term > 0:
            term_entropy += prob_class_given_term * math.log(prob_class_given_term)
    prob_term = term_probability(term, global_term_count)
    term_entropy *= prob_term

    # TODO: verify this is correct way of calculating Pr(~t) and Pr(c | ~t)
    not_term_entropy = 0.0
    for classification in classifications:
        prob_class_given_not_term = class_probability_given_not_term(classification, term, global_term_count, class_term_count)
        if prob_class_given_not_term > 0:
            not_term_entropy += prob_class_given_not_term * math.log(prob_class_given_not_term)
    prob_not_term = 1 - prob_term
    not_term_entropy *= prob_not_term

    return class_entropy + term_entropy + not_term_entropy

_all_term_occurrences_cache = None
_all_term_occurrences_in_class_cache = {}

def term_probability(term, global_term_count):
    global _all_term_occurrences_cache

    if _all_term_occurrences_cache is None:
        terms = global_term_count.keys()
        all_term_occurrences = 0.0
        for t in terms:
            all_term_occurrences += global_term_count[t]
        _all_term_occurrences_cache = all_term_occurrences

    term_occurrences = global_term_count[term]
    return term_occurrences / _all_term_occurrences_cache

def class_probability(classification, global_term_count, class_term_count):
    global _all_term_occurrences_cache
    global _all_term_occurrences_in_class_cache

    if _all_term_occurrences_cache is None:
        terms = global_term_count.keys()
        all_term_occurrences = 0.0
        for t in terms:
            all_term_occurrences += global_term_count[t]
        _all_term_occurrences_cache = all_term_occurrences

    if classification not in _all_term_occurrences_in_class_cache:
        terms = global_term_count.keys()
        all_term_occurrences_in_class = 0.0
        for t in terms:
            all_term_occurrences_in_class += class_term_count[classification][t]
        _all_term_occurrences_in_class_cache[classification] = all_term_occurrences_in_class

    return _all_term_occurrences_in_class_cache[classification] / _all_term_occurrences_cache

def class_probability_given_term(classification, term, global_term_count, class_term_count):
    term_occurrences_in_class = class_term_count[classification][term]
    term_occurrences = global_term_count[term]
    return term_occurrences_in_class / (1.0 * term_occurrences)

def term_probability_given_class(term, classification, global_term_count, class_term_count):
    global _all_term_occurrences_in_class_cache

    if classification not in _all_term_occurrences_in_class_cache:
        terms = global_term_count.keys()
        all_term_occurrences_in_class = 0.0
        for t in terms:
            all_term_occurrences_in_class += class_term_count[classification][t]
        _all_term_occurrences_in_class_cache[classification] = all_term_occurrences_in_class

    return class_term_count[classification][term] / _all_term_occurrences_in_class_cache[classification]

# TODO: cache this again
_sum_all_term_occurrences_in_class_cache = {}
def class_probability_given_not_term(classification, term, global_term_count, class_term_count):
    """
        sum_all_term_occurrences_in_class - sum_occurrences_of_term_in_class
        --------------------------------------------------------------------
        sum_all_term_occurrences - sum_occurrences_of_term
    """
    global _sum_all_term_occurrences_in_class_cache
    global _all_term_occurrences_cache

    if classification not in _sum_all_term_occurrences_in_class_cache:
        sum_all_term_occurrences_in_class = 0
        for t in class_term_count[classification].keys():
            sum_all_term_occurrences_in_class += class_term_count[classification][t]
        _sum_all_term_occurrences_in_class_cache[classification] = sum_all_term_occurrences_in_class

    sum_occurrences_of_term_in_class = class_term_count[classification][term]

    if not _all_term_occurrences_cache:
        sum_all_term_occurrences = 0
        for t in global_term_count.keys():
            sum_all_term_occurrences += global_term_count[t]
        _all_term_occurrences_cache = sum_all_term_occurrences

    sum_occurrences_of_term = global_term_count[term]

    return ((_sum_all_term_occurrences_in_class_cache[classification] - sum_occurrences_of_term_in_class) / 
            (_all_term_occurrences_cache - sum_occurrences_of_term))
