import math
import copy
import time
import random
import matplotlib.pyplot as plt
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
            if (result[0] == 0):
                count0 += 1
            elif (result[0] == 1):
                count1 += 1

        if (count0 > count1):
            self.classDist = 0
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

    #This method organizes the decission tree
    def deleteExtraChildren(self):
        result = []
        for i in range(len(self.children)):
            if(self.children[i].parent == self):
                result.append(self.children[i])
        self.children = result

def entropy(results):       #a function to calculate the entropy of a particular dataset
    entropy_value = 0.0
    rows_length = len(results)
    counted_dict = class_attrib_value_count(results)
    for value in counted_dict.keys():
        p = float(counted_dict[value])/rows_length
        if p<=0:
            p=1
        else:
            entropy_value -= (p * math.log(p,2))
    return entropy_value


def class_attrib_value_count(results):  # a function to give out the existing class distributions of a given dataset.
    count_dict = {}   # a dictionary to maintain count of each attribute value
    for row in results:
        value = row[0]
        if value in count_dict:    # if value is already in dict, increment it
            count_dict[value] += 1
        else:
            count_dict[value] = 1   # else assign its count as zero
    return count_dict

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

#This method recursively builds a decission tree for a given dataset , feature list and a  depth
def buildTree(results,totalDepth,featureList,initialDepth,parent = None):
    print "entering buildTree"
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

#This method loads the data from the files
def load_data(fileName):
    data = []
    file = open(fileName, "r")
    lines = file.readlines()
    for line in lines[1:]:
        line = line.strip()
        line = line.split(",")
        for l in range(len(line)):
            line[l]=int(line[l])
            data.append(line)
    file.close()
    return data

#This method Sets a depth value
def setDepth(value):
    totalDepth = int(value)
    return totalDepth

#Finds the accuracy
def calculate_accuracy(incorrectly_classified,correctly_classified):

    print("\n\n\nIncorrectly classified= " + str(incorrectly_classified) + "\t\t Correctly classified= " + str(correctly_classified)+"\n")
    accuracy = float(correctly_classified) / (correctly_classified + incorrectly_classified)
    print("\nAccuracy for a depth of " + str(totalDepth) + " is " + str(accuracy*100)+" %"+"\n")
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

#This method prints the decission tree
def printTree(node,num=0):
    if(node.isLeaf == False):
        for child in node.children:
            val = child.results[0][node.col]
            for i in range(num):
                print "\t",
            print "if(a"+str(node.col)+"=="+str(val)+"):",
            if(child.isLeaf == False):
                print "\n"
                printTree(child,num+1)
            else:
                print "class Distribution="+str(child.classDist)+" , Number of records="+str(len(child.results))

def learn_bagged(tdepth, numbags):
    pass;


def learn_boosted(tdepth, numtrees):
    pass;

def selectTempData(fullData):
    print "entering selectTempData method"
    tempData = []
    fullDataCopy = copy.deepcopy(fullData)
    for i in range(int(len(fullData) * 0.8)):
        x=random.choice(fullDataCopy)
        #print x
        fullDataCopy.remove(x)
        tempData.append(x)
    print "returning tempData"
    return tempData


if __name__ == '__main__':
    time1 = time.time()
    # entype = sys.argv[1];
    entype = "bag"
    # tdepth = int(sys.arg[2]);
    tdepth = 3
    nummodels = 2
    # datapath = sys.argv[4];

    training_data_file_location = "agaricuslepiotatrain1.csv"
    test_data_file_location = "agaricuslepiotatest1.csv"

    fullData = []
    testData = []

    # @todo: edit the below featureList
    # 0 to 126 except 21
    featureList = []
    for i in range(127):
        featureList.append(i)
    featureList.remove(21)

    totalDepth = setDepth(tdepth)

    fullData = load_data(training_data_file_location)
    testData = load_data(test_data_file_location)
    print "Data loaded successfully"

    head = []

    # Check which type of ensemble is to be learned
    if entype == "bag":
        # Learned the bagged decision tree ensemble
        #learn_bagged(tdepth, nummodels);
        for i in range(nummodels):
            #tempData = selectTempData(fullData)
            head.append(buildTree(fullData, totalDepth, featureList, 1))

        # print "************************ DECISSION TREE STARTS *************************"
        # printTree(head)
        # print "************************ DECISSION TREE ENDS *************************"
        incorrectly_classified = 0
        correctly_classified = 0
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        predictedList=[]
        for t in testData:

            for i in range(len(head)):
                predictedList.append(head[i].classify(t))

            predicted= max(set(predictedList), key=predictedList.count)
            if (predicted != t[20]):
                incorrectly_classified += 1
            else:
                correctly_classified += 1

            if (predicted == 1 and t[0] == 1):
                tp += 1
            elif (predicted == 0 and t[0] == 1):
                fn += 1
            elif (predicted == 1 and t[0] == 0):
                fp += 1
            elif (predicted == 0 and t[0] == 0):
                tn += 1
        acc = calculate_accuracy(incorrectly_classified,correctly_classified)
        create_confusion_matrix(tp,fn,fp,tn)

    time2 = time.time()
    print "Time taken="+str(time2-time1)
