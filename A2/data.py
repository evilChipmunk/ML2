
import numpy as np
from numpy import array
import pandas as pd 
 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold 
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.decomposition import PCA
import util


# rows = 1000
rows = 10000
# rows = 100000
# rows = 1000000

class DataPackage:
    def __init__(self):
        self.features = None
        self.predictions = None 
        self.allData = None 
        self.xTrain = None 
        self.xTrain = None 
        self.yTrain = None
        self.yTest = None 
 

baseData = 'Data\\' 

def readData(dataType, columns=None):
    if columns:
       df = pd.read_csv(baseData + dataType + '.csv' , usecols=columns) 
    df = pd.read_csv(baseData + dataType + '.csv') 
    sampleRows = rows
    if sampleRows > df.shape[0]:
        sampleRows = df.shape[0]
    return df.sample(sampleRows, replace=False, random_state=util.randState)

def createData(dataType='Adult', resampleData=False, stratify=True):

    data = None
    if str.lower(dataType) == 'heart':
        data = createHeartData(dataType)
    elif dataType == 'los':
        data = createLOSData('los2')
    else:
        data = createAdultData(dataType)

    if resampleData:
        dat = data[2]
        label = dat._values[:,-1]
        classCount = []
        classes = np.unique(label)
        for c in classes:
            classCount.append([c, label[label == c].shape[0]])
        classCount = sorted(classCount, key= lambda x: x[0])
        maxLabel = classCount[-1]
        maxClass = maxLabel[0]
        maxCount = int(maxLabel[1])
        # maxData = dat.loc[dat['LOS'] == maxClass]
        maxData = dat[dat.iloc[:, -1] == maxClass] 

        samples = []
        for c in classCount:
            minClass = c[0]
            if minClass != maxClass:
                # minData = dat.loc[dat['LOS'] == minClass] 
                minData = dat[dat.iloc[:, -1] == minClass] 
                sampled = resample(minData, replace=True, n_samples= maxCount, random_state=util.randState) 
                samples.append(sampled)
 
        for sample in samples:
            maxData = pd.concat([maxData, sample]) 
 
 
        label = maxData._values[:,-1]
        classCount = []
        classes = np.unique(label)
        for c in classes:
            classCount.append([c, label[label == c].shape[0]])
        data = (data[0], data[1], maxData)


    package = DataPackage()
    package.features = data[0]
    package.predictions = data[1]
    package.allData = data[2]
  
    xTrain, xTest, yTrain, yTest = split(package.features, package.predictions, stratify)
    package.xTrain = xTrain
    package.xTest = xTest
    package.yTrain = yTrain
    package.yTest = yTest

    (unique, counts) = np.unique(yTrain, return_counts=True)
    total = counts.sum()

    print('Train features {0} -\t count: {1}'.format(xTrain.shape[1], xTrain.shape[0]))
    print('Train classes count: {0}'.format(unique.shape))
    for i in range(counts.shape[0]):
        count = counts[i]
        print('Train class {0}: -\t {1}  %{2}'.format(unique[i], count, count/total))


    # print('{0} shape: {1}'.format(dataType, data[2].shape))
    # print(data[2].head()) 
    return package 



def createHeartData(dataType):
    # 1. #3 (age) 
    # 2. #4 (sex) 
    # 3. #9 (cp) 
    # 4. #10 (trestbps) 
    # 5. #12 (chol) 
    # 6. #16 (fbs) 
    # 7. #19 (restecg) 
    # 8. #32 (thalach) 
    # 9. #38 (exang) 
    # 10. #40 (oldpeak) 
    # 11. #41 (slope) 
    # 12. #44 (ca) 
    # 13. #51 (thal) 
    # 14. #58 (num) (the predicted attribute) 
 
    # 3 age: age in years 
    # 4 sex: sex (1 = male; 0 = female) 
    # 9 cp: chest pain type 
    # -- Value 1: typical angina 
    # -- Value 2: atypical angina 
    # -- Value 3: non-anginal pain 
    # -- Value 4: asymptomatic 
    # 10 trestbps: resting blood pressure (in mm Hg on admission to the hospital) 
    # 12 chol: serum cholestoral in mg/dl 
    # 16 fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false) 
    # 19 restecg: resting electrocardiographic results 
    # -- Value 0: normal 
    # -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) 
    # -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria  
    # 32 thalach: maximum heart rate achieved 
    # 38 exang: exercise induced angina (1 = yes; 0 = no) 
    # 40 oldpeak = ST depression induced by exercise relative to rest 
    # 41 slope: the slope of the peak exercise ST segment 
    # -- Value 1: upsloping 
    # -- Value 2: flat 
    # -- Value 3: downsloping 
    # 44 ca: number of major vessels (0-3) colored by flourosopy 
    # 51 thal: 3 = normal; 6 = fixed defect; 7 = reversable defect 
    # 58 num: diagnosis of heart disease (angiographic disease status) 
    # -- Value 0: < 50% diameter narrowing 
    # -- Value 1: > 50% diameter narrowing  
    columns = ['age', 'sex', 'cp','trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    columns = None
    df = readData(dataType, columns)  
    df = df.fillna(0)  

    
    columns_to_encode = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target' ]
    columns_to_scale  = ['age',  'trestbps', 'chol', 'thalach', 'oldpeak']
 
    # Instantiate encoder/scaler
    scaler = StandardScaler()
    ohe    = OneHotEncoder(sparse=False) 
    scaled_columns  = scaler.fit_transform(df[columns_to_scale]) 
    encoded_columns =    ohe.fit_transform(df[columns_to_encode]) 
    processed_data = np.concatenate([scaled_columns, encoded_columns], axis=1) 
    features = pd.DataFrame(processed_data[:, 0:-2])
    predictions = pd.DataFrame(processed_data[:, -1])    
    allData = pd.DataFrame(processed_data) 

    pca = PCA(n_components=15) 
    features = pca.fit(features).transform(features)
    print('PCA explained for {0}: {1}'.format(dataType, pca.explained_variance_ratio_.sum()))
    print(pca.explained_variance_ratio_)

    allData = np.concatenate([features, predictions], axis=1)

    return (features, predictions, allData)



    # features = df.iloc[:, 0:-1]
    # predictions = df.iloc[:, -1] 
  
    # features = pd.get_dummies(features, drop_first=True)
  
    # allData = features.copy()  
    
    # idx = allData.shape[1]
    # allData.insert(loc=idx, column='AtRisk', value=predictions.values)

    # return (features, predictions, allData)

def createAdultData(dataType):

    # >50K, <=50K.

    # age: continuous.
    # workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    # fnlwgt: continuous.
    # education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    # education-num: continuous.
    # marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
    # occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
    # relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    # race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    # sex: Female, Male.
    # capital-gain: continuous.
    # capital-loss: continuous.
    # hours-per-week: continuous.
    # native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.


    columns_to_encode = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    columns_to_scale  = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    # age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country 


    columns = ['age'
                ,'workclass'
                ,'fnlwgt'
                ,'education'
                ,'education-num'
                ,'marital-status'
                ,'occupation'
                ,'relationship'
                ,'race'
                ,'sex'
                ,'capital-gain'
                ,'capital-loss'
                ,'hours-per-week'
                ,'native-country'
                , 'IncomeType']
  
    columns.remove('education')
    columns.remove('native-country')
    columns_to_encode.remove('education')
    columns_to_encode.remove('native-country')
    df = readData(dataType, columns = columns) 
    # df = df.fillna(0)  
    # features = pd.get_dummies(features, drop_first=True)
  
    # Instantiate encoder/scaler
    scaler = StandardScaler()
    ohe    = OneHotEncoder(sparse=False)

    # Scale and Encode Separate Columns
    scaled_columns  = scaler.fit_transform(df[columns_to_scale]) 
    encoded_columns =    ohe.fit_transform(df[columns_to_encode])

    # Concatenate (Column-Bind) Processed Columns Back Together
    processed_data = np.concatenate([scaled_columns, encoded_columns], axis=1)
 
    # features = processed_data.iloc[:, 0:-2]
    # predictions = processed_data.iloc[:, -1] 
     
    features = pd.DataFrame(processed_data[:, 0:-2])
    predictions = pd.DataFrame(processed_data[:, -1]) 
    # allData = features.copy()
    # allData['IncomeType'] = predictions   
    allData = pd.DataFrame(processed_data)
    # print(features.cov())
    # print(features.corr())



    
    # b = np.var(features)
    # print(b)
    pca = PCA(n_components=15)
    pca = PCA(n_components=20)
    features = pca.fit(features).transform(features)
    print('PCA explained for {0}: {1}'.format(dataType, pca.explained_variance_ratio_.sum()))
    print(pca.explained_variance_ratio_)

    allData = np.concatenate([features, predictions], axis=1)

    return (features, predictions.values, allData)

 
def createLOSData(dataType):
    cols = []
    cols.append('rcount')       #1.6417
    cols.append('number_of_issues') #0.791839
    cols.append('hematocrit')	#0.212729
    cols.append('creatinine')	#0.203081
    cols.append('bmi')	        #0.201842
    cols.append('glucose')      #0.201561
    cols.append('sodium')	    #0.197752
    cols.append('pulse')        #0.195297
    cols.append('respiration')	#0.183819
    # cols.append('gender')
    # cols.append('dialysisrenalendstage')
    # cols.append('asthma')
    # cols.append('irondef') 
    # cols.append('pneum')
    # cols.append('substancedependence')
    # cols.append('psychologicaldisordermajor')
    # cols.append('depress')
    # cols.append('psychother')
    # cols.append('fibrosisandother')
    # cols.append('malnutrition')
    # cols.append('hemo')
    # cols.append('neutrophils')
    # cols.append('bloodureanitro')
    # cols.append('secondarydiagnosisnonicd9')
    # cols.append('facid')
    cols.append('lengthofstay')

    df = readData(dataType, cols)
    # df = pd.read_csv('data/los2.csv', usecols=cols)
    df = df.fillna(0) 

    features = df.iloc[:, 0:-1]
    predictions = df.iloc[:, -1]
    preds = np.empty_like(predictions, dtype=np.object)
    preds[predictions <= 3] = 'short'
    preds[np.logical_and(predictions > 3, predictions <= 6)] = 'med'
    preds[predictions > 6] = 'long'


    #limited binary prediction
    preds = np.empty_like(predictions, dtype=np.object)
    preds[predictions <= 3] = 'short'
    preds[predictions > 3] = 'long' 

    predictions = pd.DataFrame({'LOS': preds})
    # predictions = pd.DataFrame(data=preds,    # values
    #                 index=preds[0],    # 1st column as index
    #                 columns=['LOS'])  # 1st row as the column names
 

    features = features.drop(columns=['lengthofstay']) 
    features = pd.get_dummies(features, drop_first=True)

    # df[['hematocrit', 'neutrophils', 'sodium', 'glucose',
    #     'bloodureanitro', 'creatinine', 'bmi', 'pulse', 'respiration']] = df[['hematocrit', 'neutrophils', 'sodium', 'glucose',
    #                                                                           'bloodureanitro', 'creatinine', 'bmi', 'pulse', 'respiration']].apply(zscore)
    # df[['hematocrit', 'sodium', 'glucose', 'creatinine', 'bmi', 'pulse', 'respiration']] = df[['hematocrit', 'sodium', 'glucose',
    #                                                                         'creatinine', 'bmi', 'pulse', 'respiration']].apply(zscore)

    # print(features.describe())
    # features = features.apply(zscore)  # Normalization
    # print(features.describe())
    allData = features.copy()
    # allData['LOS'] = predictions.values

    idx = allData.shape[1]
    allData.insert(loc=idx, column='LOS', value=predictions.values)

    # print(allData.head())
    return (features, predictions, allData)

def createData1():
    cols = []
    cols.append('rcount')
    cols.append('gender')
    cols.append('dialysisrenalendstage')
    cols.append('asthma')
    cols.append('irondef')
    cols.append('pneum')
    cols.append('substancedependence')
    cols.append('psychologicaldisordermajor')
    cols.append('depress')
    cols.append('psychother')
    cols.append('fibrosisandother')
    cols.append('malnutrition')
    cols.append('hemo')
    cols.append('hematocrit')
    cols.append('neutrophils')
    cols.append('sodium')
    cols.append('glucose')
    cols.append('bloodureanitro')
    cols.append('creatinine')
    cols.append('bmi')
    cols.append('pulse')
    cols.append('respiration')
    cols.append('secondarydiagnosisnonicd9')
    # cols.append('facid')
    cols.append('lengthofstay')

    df = pd.read_csv('data/los.csv', usecols=cols)
    df = df.fillna(0)

    features = df.iloc[:, 0:-1]
    predictions = df.iloc[:, -1]

    features = pd.get_dummies(features, drop_first=True)

    df[['hematocrit', 'neutrophils', 'sodium', 'glucose',
        'bloodureanitro', 'creatinine', 'bmi', 'pulse', 'respiration']] = df[['hematocrit', 'neutrophils', 'sodium', 'glucose',
                                                                              'bloodureanitro', 'creatinine', 'bmi', 'pulse', 'respiration']].apply(zscore)

    print(features.describe())
    features = features.apply(zscore)  # Normalization
    print(features.describe())

    return (features, predictions)


def split(features, predictions, stratify):
    testSize = .20 
    if (stratify):  
        xTrain, xTest, yTrain, yTest = train_test_split(
            features, predictions, test_size=testSize, random_state=util.randState, stratify=predictions)
    else:
        xTrain, xTest, yTrain, yTest = train_test_split(
            features, predictions, test_size=testSize, random_state=util.randState)
        
    
    return (xTrain, xTest, yTrain, yTest)
 