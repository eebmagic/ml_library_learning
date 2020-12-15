#utility functions for neural net project
import random
def getNNPenData(fileString="datasets/pendigits.txt", limit=100000):
    """
    returns limit # of examples from penDigits file
    """
    examples=[]
    data = open(fileString)
    lineNum = 0
    for line in data:
        inVec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        outVec = [0,0,0,0,0,0,0,0,0,0]                      #which digit is output
        count=0
        for val in line.split(','):
            if count==16:
                outVec[int(val)] = 1
            else:
                inVec[count] = int(val)/100.0               #need to normalize values for inputs
            count+=1
        examples.append((inVec,outVec))
        lineNum += 1
        if (lineNum >= limit):
            break
    return examples

def getList(num,length):
    list = [0]*length
    list[num-1] = 1
    return list
    
def getNNCarData(fileString ="datasets/car.data.txt", limit=100000 ):
    """
    returns limit # of examples from file passed as string
    """
    examples=[]
    attrValues={}
    data = open(fileString)
    attrs = ['buying','maint','doors','persons','lug_boot','safety']
    attr_values = [['vhigh', 'high', 'med', 'low'],
                 ['vhigh', 'high', 'med', 'low'],
                 ['2','3','4','5more'],
                 ['2','4','more'],
                 ['small', 'med', 'big'],
                 ['high', 'med', 'low']]
    
    attrNNList = [('buying', {'vhigh' : getList(1,4), 'high' : getList(2,4), 'med' : getList(3,4), 'low' : getList(4,4)}),
                 ('maint',{'vhigh' : getList(1,4), 'high' : getList(2,4), 'med' : getList(3,4), 'low' : getList(4,4)}),
                 ('doors',{'2' : getList(1,4), '3' : getList(2,4), '4' : getList(3,4), '5more' : getList(4,4)}),
                 ('persons',{'2' : getList(1,3), '4' : getList(2,3), 'more' : getList(3,3)}),
                 ('lug_boot',{'small' : getList(1,3),'med' : getList(2,3),'big' : getList(3,3)}),
                 ('safety',{'high' : getList(1,3), 'med' : getList(2,3),'low' : getList(3,3)})]

    classNNList = {'unacc' : [1,0,0,0], 'acc' : [0,1,0,0], 'good' : [0,0,1,0], 'vgood' : [0,0,0,1]}
    
    for index in range(len(attrs)):
        attrValues[attrs[index]]=attrNNList[index][1]

    lineNum = 0
    for line in data:
        inVec = []
        outVec = []
        count=0
        for val in line.split(','):
            if count==6:
                outVec = classNNList[val[:val.find('\n')]]
            else:
                inVec.append(attrValues[attrs[count]][val])
            count+=1
        examples.append((inVec,outVec))
        lineNum += 1
        if (lineNum >= limit):
            break
    random.shuffle(examples)
    return examples


def buildExamplesFromPenData(size=10000):
    """
    build Neural-network friendly data struct
            
    pen data format
    16 input(attribute) values from 0 to 100
    10 possible output values, corresponding to a digit from 0 to 9

    """
    if (size != 10000):
        penDataTrainList = getNNPenData("datasets/pendigitsTrain.txt",int(.8*size))
        penDataTestList = getNNPenData("datasets/pendigitsTest.txt",int(.2*size))
    else :    
        penDataTrainList = getNNPenData("datasets/pendigitsTrain.txt")
        penDataTestList = getNNPenData("datasets/pendigitsTest.txt")
    return penDataTrainList, penDataTestList


def buildExamplesFromCarData(size=200):
    """
    build Neural-network friendly data struct
            
    car data format
    | names file (C4.5 format) for car evaluation domain

    | class values - 4 value output vector

    unacc, acc, good, vgood

    | attributes

    buying:   vhigh, high, med, low.
    maint:    vhigh, high, med, low.
    doors:    2, 3, 4, 5more.
    persons:  2, 4, more.
    lug_boot: small, med, big.
    safety:   low, med, high.
    """
    carData = getNNCarData()
    carDataTrainList = []
    for cdRec in carData:
        tmpInVec = []
        for cdInRec in cdRec[0] :
            for val in cdInRec :
                tmpInVec.append(val)
        #print "in :" + str(cdRec) + " in vec : " + str(tmpInVec)
        tmpList = (tmpInVec, cdRec[1])
        carDataTrainList.append(tmpList)
    #print "car data list : " + str(carDataList)
    random.shuffle(carDataTrainList)
    carDataTestList = carDataTrainList[-size:]
    carDataTrainList = carDataTrainList[:-size]
    return carDataTrainList, carDataTestList
    

def buildPotentialHiddenLayers(numIns, numOuts):
    """
    This builds a list of lists of hidden layer layouts
    numIns - number of inputs for data
    some -suggestions- for hidden layers - no more than 2/3 # of input nodes per layer, and
    no more than 2x number of input nodes total (so up to 3 layers of 2/3 # ins max
    """
    resList = []
    tmpList = []
    maxNumNodes = max(numOuts+1, 2 * numIns)
    if (maxNumNodes > 15):
        maxNumNodes = 15

    for lyr1cnt in range(numOuts,maxNumNodes):
        for lyr2cnt in range(numOuts-1,lyr1cnt+1):
            for lyr3cnt in range(numOuts-1,lyr2cnt+1):
                if (lyr2cnt == numOuts-1):
                    lyr2cnt = 0
                
                if (lyr3cnt == numOuts-1):
                    lyr3cnt = 0
                tmpList.append(lyr1cnt)
                tmpList.append(lyr2cnt)
                tmpList.append(lyr3cnt)
                resList.append(tmpList)
                tmpList = []
    return resList


def buildInputs(sampleset, instring):
    outlist = [0] * len(sampleset)
    ind = sampleset.index(instring)
    outlist[ind] = 1

    return outlist

def getNNCensusData(fileString):
    workClasses = ["?", "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]
    educationClasses = ["?", "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc",
        "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"]
    mariageClasses = ["?", "Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
    occupationClasses = ["?", "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty",
        "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]
    relationshipClasses = ["?", "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
    raceClasses = ["?", "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
    countryClasses = ["?", "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan",
        "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland",
        "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand",
        "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]

    with open(fileString) as file:
        content = file.read().strip()
    lines = content.split('\n')
    cells = [line.split(', ') for line in lines]

    allAges = [int(x[0]) for x in cells]
    allFnlwgt = [int(x[2]) for x in cells]
    allEducation = [int(x[4]) for x in cells]
    allGains = [int(x[10]) for x in cells]
    allLosses = [int(x[11]) for x in cells]

    minAge, maxAge = min(allAges), max(allAges)
    minFnlwgt, maxFnlwgt = min(allFnlwgt), max(allFnlwgt)
    minEducation, maxEducation = min(allEducation), max(allEducation)
    minGain, maxGain = min(allGains), max(allGains)
    minLosses, maxLosses = min(allLosses), max(allLosses)


    examples = []
    for line in lines:
        values = [x.strip() for x in line.split(',')]

        # age: continuous.
        age = [(int(values[0]) - minAge) / maxAge]
        # workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
        # workclass = workClasses.index(values[1])
        workclass = buildInputs(workClasses, values[1])
        # fnlwgt: continuous.
        fnlwgt = [(int(values[2]) - minFnlwgt) / maxFnlwgt]
        # education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
        # education = educationClasses.index(values[3])
        education = buildInputs(educationClasses, values[3])
        # education-num: continuous.
        educationNum = [(int(values[4]) - minEducation) / maxEducation]
        # marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
        # maritalStatus = mariageClasses.index(values[5])
        maritalStatus = buildInputs(mariageClasses, values[5])
        # occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
        # occupation = occupationClasses.index(values[6])
        occupation = buildInputs(occupationClasses, values[6])
        # relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
        # relationship = relationshipClasses.index(values[7])
        relationship = buildInputs(relationshipClasses, values[7])
        # race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
        # race = raceClasses.index(values[8])
        race = buildInputs(raceClasses, values[8])
        # sex: Female, Male.
        # sex = ["Female", "Male"].index(values[9])
        sex = buildInputs(["Female", "Male"], values[9])
        # capital-gain: continuous.
        capitalGain = [(int(values[10]) - minGain) / maxGain]
        # capital-loss: continuous.
        capitalLoss = [(int(values[11]) - minLosses) / maxLosses]
        # hours-per-week: continuous.
        hoursPerWeek = [int(values[12]) / 24]
        # native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
        # nativeCountry = countryClasses.index(values[13])
        nativeCountry = buildInputs(countryClasses, values[13])

        outputs = [["<=50K", ">50K"].index(values[14])]
        inputs = age + workclass + fnlwgt + education + educationNum + maritalStatus + occupation + relationship + race + sex + capitalGain + capitalLoss + hoursPerWeek + nativeCountry

        point = (inputs, outputs)
        examples.append(point)

    return examples

def buildExamplesFromCensus():
    censusTrainSet = getNNCensusData('datasets/census/adult.data')
    censusTestSet = getNNCensusData('datasets/census/adult.test')

    return censusTrainSet, censusTestSet
