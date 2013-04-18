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
        sys.stderr.write("Usage: python generate_examples.py <user-folder> <info-gain (1), doc frequency (2)> <percent-reduction> <tf-idf (1), tf (2), or boolean (3)> <optional output-name>\n")
        return

    user_folder_uri = sys.argv[1]

    reduce_using = "information gain"
    if int(sys.argv[2]) == 2:
        reduce_using = "document frequency"

    reduce_features_by = float(sys.argv[3])

    feature_measure_method = "tfidf"
    if int(sys.argv[4]) == 2:
        feature_measure_method = "tf"
    elif int(sys.argv[4]) == 3:
        feature_measure_method = "boolean"

    if len(sys.argv) == 6:
        output_file = "arff_files/" + sys.argv[5] + "_out.arff"
    else:
        output_file = "arff_files" + user_folder_uri + "_out.arff"

    print "Writing to "+output_file        

    data_initializer = DataInitializer(user_folder_uri, reduce_features_by, reduce_using)
    example_type, examples = data_initializer.preprocess(feature_measure_method)


    
    with open(output_file, 'w') as example_file:
        # write the metadata / schema
        example_file.write("@relation %s\n\n" % user_folder_uri)

        example_file.write("@attribute __month__ {Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec}\n")
        example_file.write("@attribute __time-of-day__ {work,evening,night,morning}\n")
        for i in range(2, len(example_type.features)):
            feature = example_type.features[i]
            if feature_measure_method == "tfidf" or feature_measure_method == "tf":
                example_file.write("@attribute %s real\n" % re.sub("\W", "_", feature))
            elif feature_measure_method == "boolean":
                example_file.write("@attribute %s {1,0}\n" % re.sub("\W", "_", feature))
        example_file.write("@attribute folderClassification {")
        for classification in example_type.classifications:
            example_file.write(re.sub("\W", "_", classification) + ",")
        example_file.write("\b}\n\n")

        example_file.write("@data\n")

        # write each example
        for example in examples:
            example_file.write(arff_format(example) + '\n')

main()
