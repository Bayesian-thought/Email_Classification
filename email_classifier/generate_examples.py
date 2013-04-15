import sys
import re
from dataset import DataInitializer

def arff_format(example):
    rep = ""
    for feature_value in example.input_vector:
        rep += "%s," % feature_value
    rep += "%s" % re.sub("\W", "_", example.output)
    return rep

def main():
    if len(sys.argv) < 4:
        sys.stderr.write("Usage: python generate_examples.py <user-folder> <percent-reduction> <tf-idf (1) or tf (2)>\n")
        return

    user_folder_uri = sys.argv[1]
    reduce_features_by = float(sys.argv[2])
    feature_measure_method = "tfidf"
    if int(sys.argv[3]) == 2:
        feature_measure_method = "tf"
    data_initializer = DataInitializer(user_folder_uri, reduce_features_by)
    example_type, examples = data_initializer.preprocess(feature_measure_method)

    with open(user_folder_uri + "_examples.arff", 'w') as example_file:
        # write the metadata / schema
        example_file.write("@relation %s\n\n" % user_folder_uri)

        example_file.write("@attribute __month__ {Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec}\n")
        example_file.write("@attribute __time-of-day__ {work,evening,night,morning}\n")
        for i in range(2, len(example_type.features)):
            feature = example_type.features[i]
            if feature.startswith("__name__"):
                example_file.write("@attribute %s {True,False}\n" % re.sub("\W", "_", feature))
            else:
                example_file.write("@attribute %s real\n" % re.sub("\W", "_", feature))

        example_file.write("@attribute folder {")
        for classification in example_type.classifications:
            example_file.write(re.sub("\W", "_", classification) + ",")
        example_file.write("\b}\n\n")

        example_file.write("@data\n")

        # write each example
        for example in examples:
            example_file.write(arff_format(example) + '\n')

main()
