import math
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
def populatedInitialData(fileName,fullData):
    file = open(fileName, "r")
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        line = line.split(" ")
        for l in range(len(line)-1):
            line[l]=int(line[l])
        fullData.append(line)
    file.close()

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

if __name__ == '__main__':
    accuracy = []
    depth = []
    curve = {}
    print("\nWelcome to the decision tree classifier implementation!")
    user_choice = "Yes"
    print("\n")
    training_data_file_location = raw_input("Please give the name of the training file to read:\n")
    print("\n")
    test_data_file_location = raw_input("Please give the name of the test file to read:\n")
    while(True):
        if(user_choice == "Yes" or user_choice == "yes"):
            fullData = []
            testData = []

            featureList = [1, 2, 3, 4, 5, 6]
            print("\n")
            dp = raw_input("Enter the depth of the tree: ")
            print("\n")
            totalDepth = setDepth(dp)
            depth.append(totalDepth)
            populatedInitialData(training_data_file_location, fullData)
            populatedInitialData(test_data_file_location, testData)

            head = buildTree(fullData, totalDepth, featureList, 1)
            print "************************ DECISSION TREE STARTS *************************"
            printTree(head)
            print "************************ DECISSION TREE ENDS *************************"
            incorrectly_classified = 0
            correctly_classified = 0
            tp = 0
            tn = 0
            fp = 0
            fn = 0

            for t in testData:
                predicted = head.classify(t)
                if (predicted != t[0]):
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
            accuracy.append(acc)
            curve[totalDepth]=acc
            create_confusion_matrix(tp,fn,fp,tn)

            user_choice = raw_input("\nDo you want to continue?(Yes/No)\n")
        else:
            break
    user_graph_choice = raw_input("\nDo you want a plot of the Depth vs Accuracy? (Yes,No)\n")
    if(user_graph_choice == "Yes" or user_graph_choice == "yes"):

        for key in sorted(curve.keys()):
            pass
        if(len(depth) and len(accuracy) >= 2):
            plt.plot(list(curve.keys()),list(curve.values()))
            plt.xlabel("Depth")
            plt.ylabel("Accuracy")
            plt.title("Plot showing Depth vs Accuracy for decision tree classifier on monks train and test set")
            plt.show()
            print("\nThank you for using the decision tree classifer!")
        else:
            print("\nEnter at least two values of depth and accuracy to generate a graph")
    else: print("\nThank you for using the decision tree classifer!")
