#!usr/bin/python
#! python2
#program to implement bagging and boosting
import random
import os, sys
import math
import math
import copy

#START OF CLASS DECISION NODE
class Decision_node:								# class to represent each node in the tree
    def __init__(self, results=None,depthLevel = 1,col=None,values=None,children=[],hasChildren = False):  #initialize each node in the decision tree
        self.results = results          # a list of lists to store the resulting rows
        self.col = col                  # a variable to store the column value of the attribute to be split on
        self.children = children        # a list containing the children to each node
        self.depthLevel = depthLevel    # height till which the tree has to be constructed
        self.isLeaf = False             # a variable to test if the node is leaf
        self.parent = None              # a variable to keep track of the parent of the current node
        self.classDist = None           # a variable to give out the class distribution of a particular node
        self.colValues = None

    #This method splits given results set based on a feature
    def splitData(self):
        resultSet = {} # a dictionary to store the rows associated to each attribute value

        for result in self.results:
            if result[self.col] not in resultSet:
                resultSet[result[self.col]] = [result]
            else:
                resultSet[result[self.col]].append(result)

        return resultSet

    #This method sets the leaf nodes of decission tree to some class : (0,1)
    def setClassDist(self): # a method to store the class distribution for a node based on majority values
        results = self.results

        count0 = 0
        count1 = 0
        for result in results:
            if (result[20] == 0):
                count0 += 1
            elif (result[20] == 1):
                count1 += 1

        if (count0 > count1):
            self.classDist = 0
        else:
            self.classDist = 1

    def setClassDistBoost(self):  # a method to store the class distribution for a node based on majority values
        results = self.results

        count0 = 0
        count1 = 0
        for result in results:
            if (result[20] == -1):
                count0 += 1
            elif (result[20] == 1):
                count1 += 1

        if (count0 > count1):
            self.classDist = -1
        else:
            self.classDist = 1

    #This method classifies a given test record to either : (0/1)
    def classify(self,testRecord):
        if(self.isLeaf):
            return self.classDist
        else:
            col = self.col
            for child in self.children:
                if(child.results[0][col] == testRecord[col]):
                    return child.classify(testRecord)

    #This method organizes the decision tree
    def deleteExtraChildren(self):
        result = []
        for i in range(len(self.children)):
            if(self.children[i].parent == self):
                result.append(self.children[i])
        self.children = result

#END OF CLASS DECISION NODE

def class_attrib_value_count(results):  # a function to give out the existing class distributions of a given dataset.
    count_dict = {}   # a dictionary to maintain count of each attribute value
    for row in results:
        value = row[20]
        if value in count_dict:    # if value is already in dict, increment it
            count_dict[value] += 1
        else:
            count_dict[value] = 1   # else assign its count as zero
    return count_dict

def entropy(results):       #a function to calculate the entropy of a particular dataset
    entropy_value = 0.0
    rows_length = len(results)
    counted_dict = class_attrib_value_count(results)
    for value in counted_dict.keys():
        p = float(counted_dict[value])/rows_length
        if p<=0:
            p=float(1)/float(rows_length+2)
        else:
            entropy_value -= (p * math.log(p,2))
    return entropy_value

#This method finds if a given dataset is pure or not i.e., is it all from same class - (0/1)
def isImPure(results):
    count0=0
    count1=0
    for result in results:
        if(result[0]==0):
            count0 +=1
        elif(result[0]==1):
            count1+=1
        if(count0>0 and count1>0):
            return True
    return False


#START OF TREE BUILDING RECURSIVE FUNCTION : This method recursively builds a decision tree for a given dataset , feature list and a  depth
def buildTree(results,totalDepth,featureList,initialDepth,parent = None):

    newNode = Decision_node(results, initialDepth)
    newNode.parent = parent
    best_gain = 0
    best_attrib = None
    best_partition= None
    current_entropy = entropy(results) # find out the entropy of the new node containing the subset
    for column in featureList:
        newNode.col = column
        partitions = newNode.splitData()  # split up the node into their resulting children along with their own subsets

        new_entropy = 0.0 # set the intermediate entropy computation to zero
        for val in partitions: # loop through all the possible column values
            new_entropy = new_entropy + (entropy(partitions[val]) * (float(len(partitions[val]))/len(results)) ) # calculate the weighted entropy for that column
        information_gain = current_entropy - new_entropy
        if (information_gain > best_gain):
            best_gain = information_gain
            best_attrib = column
            best_partition = partitions

    newNode.col = best_attrib # set the column with highest information gain(best attribute) to be the splitting column
    if(newNode.depthLevel<=totalDepth and len(results)>1 and isImPure(results) and best_attrib!=None) :
        resultSet = best_partition
        newNode.colValues=resultSet.keys()
        for i in resultSet:
            x = buildTree(resultSet[i],totalDepth,featureList,initialDepth+1,newNode)
            if x.depthLevel == newNode.depthLevel+1:
                newNode.children.append(x)
    else:
        newNode.isLeaf = True
        newNode.children = []
        newNode.setClassDist()

    newNode.deleteExtraChildren()
    return newNode

#END OF TREE BUILDING RECURSIVE FUNCTION


def load_data(datapath): #load the data
    data = []
    file = open(datapath, "r")
    lines = file.readlines()
    for line in lines[1:]:
        newline = line.strip().split(",")
        newline = map(int, newline)
        data.append(newline)
    file.close()
    return data

def learn_bagged(tdepth, nummodels, datapath):
    train_data = [] #declare empty lists to store all the training records and test records
    test_data = []
    train_datapath = datapath+"/agaricuslepiotatrain1.csv"
    test_datapath = datapath + "/agaricuslepiotatest1.csv"
    train_data = load_data(train_datapath)
    print "Number of records in training set = "+str(len(train_data))
    test_data = load_data(test_datapath)
    featureList = []
    for i in range(127):
        featureList.append(i)
    featureList.remove(21)  #remove bruises-no and class label
    featureList.remove(20)
    totalDepth = tdepth

    #loop to create three bootstrap samples according to value given in nummodels
    samples = [] #contains bootstrap samples
    for i in range(1,nummodels+1):

        temp_list = []
        for k in range(int(len(train_data) * 0.8)):
            number = random.randrange(1,len(train_data))

            temp_list.append(train_data[number])

        samples.append(temp_list)

    print "Number of bags = "+str(len(samples))

    head = []

    for sample_count in range(nummodels):
        bootstrap = samples[sample_count]
        print "Number of records in the bootstrap sample = "+str(len(bootstrap))
        head.append(buildTree(bootstrap,totalDepth, featureList, 1)) # create required number of decision trees and append it to head list.
    print "Number of decision trees formed = "+str(len(head))
    incorrectly_classified = 0
    correctly_classified = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for t in test_data:
        tcopy = copy.deepcopy(t)
        tcopy[20] = None
        predictedList = []
        for i in range(len(head)):
            predictedList.append(head[i].classify(tcopy))

        predicted = max(set(predictedList),key = predictedList.count) #select the maximum predicted value
        if(predicted != t[20]):
            incorrectly_classified += 1
        else:
            correctly_classified +=1

        if (predicted == 1 and t[20] == 1):
            tp += 1
        elif (predicted == 0 and t[20] == 1):
            fn += 1
        elif (predicted == 1 and t[20] == 0):
            fp += 1
        elif (predicted == 0 and t[20] == 0):
            tn += 1
    acc = calculate_accuracy(incorrectly_classified, correctly_classified)
    create_confusion_matrix(tp, fn, fp, tn)

#Finds the accuracy
def calculate_accuracy(incorrectly_classified,correctly_classified,isPrint=True):

    if isPrint:
        print("\n\n\nIncorrectly classified= " + str(incorrectly_classified) + "\t\t Correctly classified= " + str(correctly_classified)+"\n")
    accuracy = float(correctly_classified) / (correctly_classified + incorrectly_classified)
    if isPrint:
        print("\nAccuracy for a depth of " + str(tdepth) + " is " + str(accuracy*100)+" %"+"\n")
    return accuracy

#This method prints the confusion matrix
def create_confusion_matrix(tp,fn,fp,tn):
    print "\nThe confusion matrix is as follows:\n"
    print "                                 Predicted"
    print "----"*20
    print "Actual"
    print "                 True Negative: "+str(tn),
    print "                 False Positive: "+str(fp)
    print "\n"
    print "                 False Negative: "+str(fn),
    print "                 True Positive: "+str(tp)
    print "----"*20

############################################################################################################################################

#BOOSTING SECTION

# function which return a dictionary for the weights associated to each column's values
def weights_value_counts(results):
    weight_count_dict = {}
    for row in results:
        value = row[20]
        weight = row[-1]  #consider the weights associated to each record in training set
        if value in weight_count_dict:
            weight_count_dict[value] += weight
        else:
            weight_count_dict[value] = weight
    return weight_count_dict


def entropy_boost(results):       #a function to calculate the entropy of a particular dataset
    entropy_value = 0.0
    rows_length = len(results)
    weight_count_dict =  weights_value_counts(results)
    for value in weight_count_dict.keys():
        p = float(weight_count_dict[value])/rows_length  #calculate probability with weights instead of count
        if p<=0:
            p = float(1/(rows_length+2)) #apply smoothing for p=0 cases
        else:
            entropy_value -= (p * math.log(p,2))
    return entropy_value

#build a tree using a new buildTree for boosting
def buildTree_boost(results,totalDepth,featureList,initialDepth,parent = None):

    newNode = Decision_node(results, initialDepth)
    newNode.parent = parent
    best_gain = 0
    best_attrib = None
    best_partition= None
    current_entropy = entropy_boost(results) # find out the entropy of the new node containing the subset
    for column in featureList:
        newNode.col = column
        partitions = newNode.splitData()  # split up the node into their resulting children along with their own subsets

        new_entropy = 0.0 # set the intermediate entropy computation to zero
        for val in partitions: # loop through all the possible column values
            new_entropy = new_entropy + (entropy_boost(partitions[val]) * (float(len(partitions[val]))/len(results)) ) # calculate the weighted entropy for that column
        information_gain = current_entropy - new_entropy
        if (information_gain > best_gain):
            best_gain = information_gain
            best_attrib = column
            best_partition = partitions

    newNode.col = best_attrib # set the column with highest information gain(best attribute) to be the splitting column
    if(newNode.depthLevel<=totalDepth and len(results)>1 and isImPure(results) and best_attrib!=None) :
        resultSet = best_partition
        newNode.colValues=resultSet.keys()
        for i in resultSet:
            x = buildTree_boost(resultSet[i],totalDepth,featureList,initialDepth+1,newNode)
            if x.depthLevel == newNode.depthLevel+1:
                newNode.children.append(x)
    else:
        newNode.isLeaf = True
        newNode.children = []
        newNode.setClassDistBoost()

    newNode.deleteExtraChildren()
    return newNode


def learn_boosted(tdepth, nummodels, datapath):
    train_datapath = datapath+"/agaricuslepiotatrain1.csv"
    test_datapath = datapath + "/agaricuslepiotatest1.csv"
    train_data = load_data(train_datapath)
    print "Number of records in training set = "+str(len(train_data))
    test_data = load_data(test_datapath)
    featureList = []
    for i in range(127):
        featureList.append(i)
    featureList.remove(21)
    featureList.remove(20)
    totalDepth = tdepth
    # replace the 0 in the training and test datasets by -1
    for row in train_data:
        if(row[20] == 0):
            row[20] = -1
        else:
            row[20] = 1
    for row2 in test_data:
        if(row2[20] == 0):
            row2[20] = -1
        else:
            row2[20] = 1


    #settings initial weight vector = 1/N and adding it to training data
    equal_weights = float(1) / float(len(train_data))
    for row in train_data:
        row.append(equal_weights)

    headList=[]
    for i in range(nummodels): #for each iteration, build a tree with updated weights
        head = buildTree_boost(train_data,totalDepth,featureList,1)

        incorrectly_classified = 0
        correctly_classified = 0
        predictedList = []
        observedList = []
        for t in train_data:   # run on the train data to calculate error
            predicted = head.classify(t)
            predictedList.append(predicted)
            observedList.append(t[20])
            if(predicted != t[20]):
                incorrectly_classified += 1
            else:
                correctly_classified +=1
        acc = calculate_accuracy(incorrectly_classified, correctly_classified,False)
        error = (1 - acc)
        error_compute = float(1-error)/float(error)
        alpha = 0.5 * math.log(error_compute)
        headList.append([head,alpha])
        normalization_factor = 0.0
        for row3 in train_data:
             normalization_factor += row3[-1]
        for i in range(len(train_data)):   # update the training data weights with new weights
            train_data[i][-1] = (train_data[i][-1] * math.exp((-1)* alpha *predictedList[i]*observedList[i]))/normalization_factor

    incorrectly_classified = 0
    correctly_classified = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    x=set()
    for t in test_data: #traverse a test record through all models and get the final predicted value
        tcopy = copy.deepcopy(t)
        tcopy[20] = None
        predictedList = []
        for i in range(len(headList)):
            predicted=headList[i][0].classify(tcopy)
            x.add(predicted)
            predicted *= headList[i][1] #calculate final hypothesis
            predictedList.append(predicted)

        if (sum(predictedList) < 0): #if final hypothesis value is < 0, predict -1 and 1 otherwise
            predicted=-1
        else:
            predicted = 1

        if (predicted != t[20]):
            incorrectly_classified += 1
        else:
            correctly_classified += 1

        if (predicted == 1 and t[20] == 1):
            tp += 1
        elif (predicted == -1 and t[20] == 1):
            fn += 1
        elif (predicted == 1 and t[20] == -1):
            fp += 1
        elif (predicted == -1 and t[20] == -1):
            tn += 1
    acc = calculate_accuracy(incorrectly_classified, correctly_classified)
    create_confusion_matrix(tp, fn, fp, tn)

if __name__ == "__main__":
    # The arguments to your file will be of the following form:
    # <ensemble_type> <tree_depth> <num_bags/trees> <data_set_path>
    # Ex. bag 3 10 mushrooms
    # Ex. boost 1 10 mushrooms

    # Get the ensemble type
    entype = sys.argv[1];
    #entype = "boost"
    # Get the depth of the trees
    tdepth = int(sys.argv[2])
    #tdepth = 2
    # Get the number of bags or trees
    nummodels = int(sys.argv[3]);
    #nummodels = 10
    # Get the location of the data set
    datapath = sys.argv[4];

    # Check which type of ensemble is to be learned
    if entype == "bag":
        # Learned the bagged decision tree ensemble
        learn_bagged(tdepth, nummodels, datapath);
    else:
        # Learned the boosted decision tree ensemble
        learn_boosted(tdepth, nummodels, datapath);
