import sys
import math
from collections import OrderedDict
from collections import defaultdict
import operator
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def first_pass(training_file, all_features, occurrence_of_class):
    with open(training_file, "r") as rfp:
        line = rfp.readline()

        while line:
            elements = line.strip().split() ## a list of elements in line
            
            class_name = elements[0] ## this first element is the class name
            
            if class_name not in occurrence_of_class:
                occurrence_of_class[class_name] = [1, 0.0, 0.0]
            else:
                occurrence_of_class[class_name][0] = occurrence_of_class[class_name][0] + 1
                
            list_of_features = elements[1:]

            for feature_pair in list_of_features:
            
                feature = feature_pair.split(':')[0]

                if feature not in all_features:
                    all_features[feature] = True

            line = rfp.readline()

def second_pass(training_file, training_data_structure, all_features, occurrence_of_class, 
                z_dictionary):
    with open(training_file, "r") as rfp:
        line = rfp.readline()
        
        for class_name in occurrence_of_class:
        
            if class_name not in training_data_structure:
            
                z_dictionary[class_name] = 0
                training_data_structure[class_name] = {} ## create a new dictionary
                
            for possible_feature in all_features:
                ## the key will be a list of 3 elements:
                ## occurrence, prob and log_prob
                training_data_structure[class_name][possible_feature] = [0, 0.0, 0.0]    
                
        while line:
            elements = line.strip().split() ## a list of elements in line
            
            class_name = elements[0] ## this first element is the class name
            
            list_of_features = elements[1:]
            
            for feature_pair in list_of_features:
            
                feature_occurrence_pair = feature_pair.split(':')
                
                feature = feature_occurrence_pair[0]
                
                occurrence = int(feature_occurrence_pair[1])
                
                training_data_structure[class_name][feature][0] += occurrence
                
                z_dictionary[class_name] += occurrence

            line = rfp.readline()                       
                                    
def read_training_data(training_file):
    training_data_structure = {} ## this is a dictionary of dictionaries
 
    all_features = {} ## this is a dictionary that maps all the features to True
    
    ## this is a dictionary that maps the class_name to 
    ## a list of 3 elements: first one is the total number of training instances that 
    ## have the class_name, second one is the prob, and third one is the log10 prob
    
    occurrence_of_class = {}
    
    z_dictionary = {}

    first_pass(training_file, all_features, occurrence_of_class)

    second_pass(training_file, training_data_structure, all_features, occurrence_of_class, z_dictionary)

    return training_data_structure, all_features, occurrence_of_class, z_dictionary

def count_feature(feature_dictionary, feature):
    return feature_dictionary[feature][0]
        
def output_model_file(training_data_structure, model_file, class_prior_delta, cond_prob_delta,
                      all_features, occurrence_of_class, z_dictionary):
    with open(model_file, "w") as wfp:
        wfp.write('%%%%% prior prob P(c) %%%%%\n')
        
        total_class_instance_count = 0
        
        number_of_classes = len(training_data_structure)
        
        for class_name in training_data_structure:
                
            ## return the total number of counts 
            count = occurrence_of_class[class_name][0]
            total_class_instance_count = total_class_instance_count + count

        for class_name in training_data_structure:
            count_c_i = occurrence_of_class[class_name][0]
            prob = float(class_prior_delta + count_c_i) / float(class_prior_delta * number_of_classes + total_class_instance_count)
            log_prob = math.log10(prob)
            wfp.write(class_name + '\t' + str(prob) + '\t' + str(log_prob) + '\n')
            
            occurrence_of_class[class_name][1] = prob
            occurrence_of_class[class_name][2] = log_prob
            
        wfp.write('%%%%% conditional prob P(f|c) %%%%%\n')
        
        all_features = OrderedDict(sorted(all_features.items()))
                       
        for class_name in training_data_structure:    
                
            wfp.write('%%%%% conditional prob P(f|c) c=' + class_name + ' %%%%%\n')    
            
            feature_dictionary = training_data_structure[class_name]
            
            class_count = occurrence_of_class[class_name][0]
                
            for feature in all_features:
             
                feature_count = count_feature(feature_dictionary, feature)
                
                prob = float(cond_prob_delta + feature_count) / float(cond_prob_delta * len(all_features) + z_dictionary[class_name])
                
                log_prob = math.log10(prob)
                
                wfp.write(feature + '\t' + class_name + '\t' + str(prob) + '\t' + str(log_prob) + '\n')
                
                training_data_structure[class_name][feature][1] = prob
                training_data_structure[class_name][feature][2] = log_prob
                

def write_to_sys_file(predict, i, wfp, predicted_class_name):
        
        ## adjust the ratio
        
    adjust_predict = {}
        
        ## this is the base to adjust with

    base = predict[predicted_class_name]        
        
    for class_name in predict:
        if class_name not in adjust_predict:
            adjust_predict[class_name] = predict[class_name] - base
        
    sum_prob = 0.0
        
    output_dictionary = {}

    for class_name in adjust_predict:
        sum_prob = sum_prob + pow(10, adjust_predict[class_name])

    for class_name in adjust_predict:
        prob = pow(10, adjust_predict[class_name]) / sum_prob

        if class_name not in output_dictionary:
            output_dictionary[class_name] = prob

    sorted_dictionary = sorted(output_dictionary.items(), key=operator.itemgetter(1), reverse=True)
    
    string_to_write = ''
    
    for element in sorted_dictionary:
        string_to_write = string_to_write + element[0] + '\t' + str(element[1]) + '\t'
    
    wfp.write('array:' + str(i) + '\t' + predicted_class_name + '\t' + string_to_write + '\n')
                
                        
def classify(training_data_structure, all_features, occurrence_of_class, wfp, training_file, type):
    
    wfp.write('%%%%% training data:\n')
    
    y_true = []
    y_pred = []
    
    with open(training_file, "r") as rfp:
        
        ## i is the index number
        
        for i, line in enumerate(tqdm(rfp)):           
            
            elements = line.strip().split() ## a list of elements in line
            
            claimed_class_name = elements[0] ## this first element is the claimed class name
            
            list_of_elements = elements[1:]
            
            ## map each feature to its occurrence
            
            target_feature = {}
            
            for pair in list_of_elements:
                feature_occurrence_pair = pair.split(':')
                feature = feature_occurrence_pair[0]
                occurrence = int(feature_occurrence_pair[1])
                target_feature[feature] = occurrence
                
            predicted_class_name = None
            
            ## a dictionary that maps a class_name to 
            ## its log_classify_val

            predict = {}            
                
            for class_name in training_data_structure:
            
                class_prob = 0
            
                stored_features = training_data_structure[class_name]
                
                num_times_wk_appears_in_di = 0
                
                for feature in stored_features:
                
                    if feature in target_feature:
                        num_times_wk_appears_in_di = target_feature[feature]
                    else:
                        num_times_wk_appears_in_di = 0

                    class_prob = class_prob + (num_times_wk_appears_in_di * stored_features[feature][2])

                class_prob = class_prob + occurrence_of_class[class_name][2]

                if class_name not in predict:
                    predict[class_name] = class_prob                
                        
                    
            predicted_class_name = max(predict.items(), key = operator.itemgetter(1))[0]
            y_true.append(claimed_class_name)
            y_pred.append(predicted_class_name)            
                                
            write_to_sys_file(predict, i, wfp, predicted_class_name)
            
        sorted_list = sorted(occurrence_of_class.items(), key=lambda x: x[0])

        label_list = []

        for element in sorted_list:
            label_list.append(element[0])

        cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=label_list), index=label_list, columns=label_list)

        print("Confusion matrix for the %s data:" % type)

        print("row is the truth, column is the system output\n")

        print(cm.to_string() + "\n")

        print("accuracy=%s\n\n" % (accuracy_score(y_true, y_pred)))
        
        
                                                
def main():

    ## get all the parameters from the command line

    training_file = sys.argv[1]
    testing_file = sys.argv[2]
    class_prior_delta = float(sys.argv[3])
    cond_prob_delta = float(sys.argv[4])
    model_file = sys.argv[5]
    sys_file = sys.argv[6]
    
    
    training_data_structure, all_features, occurrence_of_class, z_dictionary = read_training_data(training_file)
    
    training_data_structure = OrderedDict(sorted(training_data_structure.items()))
    
    feature_class_dictionary = output_model_file(training_data_structure, model_file, class_prior_delta, cond_prob_delta,
                                                 all_features, occurrence_of_class, z_dictionary)
    
    """
    
    print(len(all_features))                                             
                                                 
    for class_name in training_data_structure:
        print('the length of class ' + class_name + ' is ' + str(len(training_data_structure[class_name])))

        print(training_data_structure[class_name])

    """ 

    with open(sys_file, "w") as wfp:    
                               
        classify(training_data_structure, all_features, occurrence_of_class, wfp, training_file, "training")
        wfp.write('\n')
        wfp.write('\n')
        classify(training_data_structure, all_features, occurrence_of_class, wfp, testing_file, "testing")
                
    
    
if __name__ == '__main__':
    main()    
    