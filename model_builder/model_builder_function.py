import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import shapiro
from scipy.stats import pearsonr
from scipy.stats import f_oneway
#from scipy.stats import ttest_ind
#import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# Fonction Permettant de convertir en float les nombre dont le séparateur décimal est la virgule ','
# def toFloat(str0):
#     value = 0
#     if type(str0) == str:
#         elmts = str0.split(sep=',')
#         str1 = elmts[0]
#         if len(elmts) > 1 :
#             str1 += '.' + elmts[1]
#         value = float(str1)
#     else:
#         value = float(str0)
#     return value

# Fonction d'obtention d'un ratio
def getRatio(index, X_data):
    n0 = len(X_data)
    R = list()
    for i in range(n0):
        R.append(X_data[i][index])
    return R

# Récupération de la taille de l'échantillon
def build_model(data, verbose=True):
    myModel = dict()

    n = len(data)
    if(verbose):
        print("Dataset length :",n)
    
    myModel['dataLength'] = n

    # I) CONSTRUCTION DU DATASET
    ## 1) Libellé des Ratios et Fonction de lecture du dataset
    ## 2) Construction des Ratios et de la variable expliqué

    ## I.1) Libellé des Ratios
    features = ['noteSecteur','notePositionnement','noteReferences','noteActionnariat','noteGouvernance','noteAvisBanqueCentrale',
    'noteCreditCaisse','noteFEDImport','noteCMTInvestissement','noteMouvementsCompte','RN_CA','REX_AE','EBE_CA','REX_CA','FPN_TB',
    'FR_BFR','CP_AI','FPN_AIhp', 'CFN_CA', 'TN_CA', 'CharFin_EBE','CharFin_REX','DMLT_CFN','DMLT_EBE','DMLT_CP','CharFin_CA',
    'DMLT_FPN', 'CP_TB','CharFin_VA','AI_CA','CharPers_CA','D_CLT','D_STK','D_Fourn', 'FR_CA']
    myModel['features'] = features

    nRatios = len(features)
    X = np.zeros((n,nRatios))
    ylist = list()

    if(verbose):
        print("X shape :",X.shape)

    ## I.2) Construction des Ratios et de la variable expliqué
    i = 0
    for train_example in data:
        X[i][0] = train_example['noteSecteur']
        X[i][1] = train_example['notePositionnement']
        X[i][2] = train_example['noteReferences']
        X[i][3] = train_example['noteActionnariat']
        X[i][4] = train_example['noteGouvernance']
        X[i][5] = train_example['noteAvisBanqueCentrale']
        X[i][6] = train_example['noteCreditCaisse']
        X[i][7] = train_example['noteFEDImport']
        X[i][8] = train_example['noteCMTInvestissement']
        X[i][9] = train_example['noteMouvementsCompte']
        X[i][10] = train_example['resultatNet']/train_example['ca']
        X[i][11] = train_example['resultatExploitation']/train_example['actifsEconomiques']
        X[i][12] = train_example['ebe']/train_example['ca']
        X[i][13] = train_example['resultatExploitation']/train_example['ca']
        X[i][14] = train_example['fondsPropresNet']/train_example['totalBilan']
        X[i][15] = train_example['fondsRoulement']/train_example['bfr']
        X[i][16] = train_example['capitauxPermanents']/train_example['actifsImmobilises']
        X[i][17] = train_example['fondsPropresNet']/train_example['actifsImmobilisesHP']
        X[i][18] = train_example['cashFlowNet']/train_example['ca']
        X[i][19] = train_example['tresorerie']/train_example['ca']
        X[i][20] = train_example['chargesFinancieres']/train_example['ebe']
        X[i][21] = train_example['chargesFinancieres']/train_example['resultatExploitation']
        X[i][22] = train_example['dmlt']/train_example['cashFlowNet']
        X[i][23] = train_example['dmlt']/train_example['ebe']
        X[i][24] = train_example['dmlt']/train_example['capitauxPermanents']
        X[i][25] = train_example['chargesFinancieres']/train_example['ca']
        X[i][26] = train_example['dmlt']/train_example['fondsPropresNet']
        X[i][27] = train_example['capitauxPermanents']/train_example['totalBilan']
        X[i][28] = train_example['chargesFinancieres']/train_example['valeurAjoutee']
        X[i][29] = train_example['actifsImmobilises']/train_example['ca']
        X[i][30] = train_example['chargesPersonnel']/train_example['ca']
        X[i][31] = (train_example['moyenneCreances']*360)/train_example['ca']
        X[i][32] = (train_example['moyenneStocks']*360)/train_example['ca']
        X[i][33] = (train_example['moyenneDettesFournisseurs']*360)/train_example['achatsTTC']
        X[i][34] = train_example['fondsRoulement']/train_example['ca']
        ylist.append(int(train_example['y']))
        i +=1
    y = np.array(ylist)
    del(ylist)
    if(verbose):
        print("The Data set :", X)
        print('*********************************************************************************')
        print("Y values:", y)
    

    # II) DESCRIPTION DES DONNEES
    ## 1) Test de Skewness
    ## 2) Test de Kurtosis
    ## 3) Test de Normalité de Shapiro Wilk
    ## 4) Test de correlation lineaire (Statistique de pearson)
    ## 5) ANOVA à un facteur (Test de Fisher)

    ## II.1) Test de Skewness
    positiveSkew = list()
    negativeSkew = list()
    nullSkew = list()
    valuesSkew = list()
    for j in range(nRatios):
        currentSkew = skew(getRatio(j, X))
        valuesSkew.append(currentSkew)
        if currentSkew > 0:
            positiveSkew.append('R_'+str(j))
        elif currentSkew < 0:
            negativeSkew.append('R_'+str(j))
        else:
            nullSkew.append('R_'+str(j))
    
    # myModel['positiveSkew'] = positiveSkew
    # myModel['negativeSkew'] = negativeSkew
    # myModel['nullSkew'] = nullSkew
    myModel['skewness'] = valuesSkew

    if(verbose):
        print("Skewness Positif - Distribution décalée à gauche")
        print('\t'+ str(positiveSkew))
        print("Skewness Négatif - Distribution décalée à droite")
        print('\t'+ str(negativeSkew))
        print("Skewness Nul - Variable symétrique (Comme pour une distribution Normale)")
        print('\t'+ str(nullSkew))


    ## II.2) Test de Kurtosis
    positiveKurtosis = list()
    negativeKurtosis = list()
    nullKurtosis = list()
    valuesKurtosis = list()
    for j in range(nRatios):
        currentKurtosis = kurtosis(getRatio(j, X))
        valuesKurtosis.append(currentKurtosis)
        if currentKurtosis > 0:
            positiveKurtosis.append('R'+str(j+1))
        elif currentKurtosis < 0:
            negativeKurtosis.append('R'+str(j+1))
        else :
            nullKurtosis.append('R'+str(j+1))

    # myModel['positiveKurtosis'] = positiveKurtosis
    # myModel['negativeKurtosis'] = negativeKurtosis
    # myModel['nullKurtosis'] = nullKurtosis
    myModel['kurtosis'] = valuesKurtosis

    if(verbose):
        print("Kurtosis Positif - Distribution Pointue")
        print('\t'+ str(positiveKurtosis))
        print("Kurtosis Négatif - Distribution applatie")
        print('\t'+ str(negativeKurtosis))
        print("Kurtosis Nul - Distribution ni pointue ni applatie (comme une loi normale)")
        print('\t'+ str(nullKurtosis))
    
    ## II.3) Test de Normalité de Shapiro Wilk
    normaliteValues = list()

    for j in range(nRatios):
        stat, p = shapiro(getRatio(j, X))
        normaliteValues.append({"stat": stat,"p_value":p})
        if(verbose):
            print('R' + str(j+1) + ' :','stat=%.3f, p=%.3f' % (stat, p))
            if p > 0.05:
                print('\tProbablement Gaussienne')
            else:
                print('\tProbablement non Gaussienne')
    
    myModel['normalite'] = normaliteValues

    ## II.4) Test de correlation de Pearson (Test de corrélation linéaire)
    ratiosList = list(range(nRatios))
    #print(ratiosList)
    linearGroups = list()
    indexLinearGroups = list()
    pos = 0
    while ratiosList:
        #print(ratiosList)
        correlated = list()
        group = list()
        j = ratiosList[0]
        correlated.append(j)
        group.append('R'+str(j))
        ratiosList.remove(j)

        Rn = getRatio(j, X)
        for k in ratiosList:
            Rm = getRatio(k, X)
            stat, p = pearsonr(Rn, Rm)
            #print('(R' + str(j+1) + ', R' + str(k+1) + ')','stat=%.3f, p=%.3f' % (stat, p))
            if p <= 0.05:
                correlated.append(k)
                group.append('R'+str(k))
        if(verbose):
            print("Goupe ",pos)
            print("\t"+ str(group))
        pos +=1
        indexLinearGroups.append(correlated.copy())
        linearGroups.append(group)
        correlated.remove(j)
        for el in correlated:
            ratiosList.remove(el)
    
    myModel['pearson'] = indexLinearGroups


    # X_noLC = np.zeros((n,len(indexLinearGroups)))
    # for i in range(n):
    #     j = 0
    #     for grp in indexLinearGroups:
    #         X_noLC[i][j] = X[i][grp[0]]
    #         j+=1


    # Analysis of Variance Test
    def getRatioByClass(index, X_data, y):
        n0 = len(X_data)
        class0 = list()
        class1 = list()
        for i in range(n0):
            if y[i] ==1 :
                class1.append(X_data[i][index])
            else:
                class0.append(X_data[i][index])
        return(class0,class1)

    indexSelect = list()
    fischerValues = list()
    for i in range(nRatios):
        classes = getRatioByClass(i, X, y)
        stat, p = f_oneway(classes[0], classes[1])
        fischerValues.append({"stat": stat,"p_value":p})
        #stat1, p1 = ttest_ind(classes[0], classes[1])
        if(verbose):
            print('R'+str(i+1)+' stat=%.3f, p=%.3f' % (stat, p))
        #print('Student R'+str(i+1)+' stat=%.3f, p=%.3f' % (stat1, p1))
        if p > 0.05:
            if(verbose):
                print('\t Selection : NON')
        else:
            indexSelect.append(i)
            if(verbose):
                print('\t Selection : OUI')
    if(verbose):
        print('Liste des ratios sélectionnés')
        print(indexSelect)
    
    myModel['fischer'] = fischerValues
    myModel['selection'] = indexSelect

    ##Récupération des ratios sélectionnés dans une matrice
    X_select = np.zeros((n,len(indexSelect)))
    for i in range(n):
        j = 0
        for k in indexSelect:
            X_select[i][j] = X[i][k]
            j+=1
    
    # II) CONSTRUCTION DU MODEL
    ##Séparation du data set en un emsemble d'entrainement et un ensemple de test
    X_train, X_test, y_train, y_test = train_test_split(X_select, y,test_size=0.20, random_state=25)

    model = LogisticRegression(solver='newton-cg', verbose=1, max_iter = 1000, C = 1).fit(X_select,y)

    myModel['params'] = {"bias": float(model.intercept_[0]), "weights":list(model.coef_[0])}
    datasetCM = confusion_matrix(y, model.predict(X_select))
    trainsetCM = confusion_matrix(y_train, model.predict(X_train))
    testsetCM = confusion_matrix(y_test, model.predict(X_test))

    datasetP = datasetCM[1][1]/(datasetCM[1][1]+datasetCM[0][1])
    datasetRecal = datasetCM[1][1]/(datasetCM[1][1] + datasetCM[1][0])
    datasetF1 = (2*datasetP*datasetRecal)/(datasetP+datasetRecal)

    trainsetP = trainsetCM[1][1]/(trainsetCM[1][1]+trainsetCM[0][1])
    trainsetRecal = trainsetCM[1][1]/(trainsetCM[1][1] + trainsetCM[1][0])
    trainsetF1 = (2*trainsetP*trainsetRecal)/(trainsetP+trainsetRecal)

    testsetP = testsetCM[1][1]/(testsetCM[1][1]+testsetCM[0][1])
    testsetRecal = testsetCM[1][1]/(testsetCM[1][1] + testsetCM[1][0])
    testsetF1 = (2*testsetP*testsetRecal)/(testsetP+testsetRecal)

    myModel['datasetPerformance'] = {'tn': int(datasetCM[0][0]) ,'fp': int(datasetCM[0][1]),'fn': int(datasetCM[1][0]),'tp': int(datasetCM[1][1]), 'precision': datasetP, 'recal': datasetRecal, 'f1score': datasetF1}
    myModel['trainsetPerformance'] = {'tn': int(trainsetCM[0][0]) ,'fp': int(trainsetCM[0][1]),'fn': int(trainsetCM[1][0]),'tp': int(trainsetCM[1][1]), 'precision': trainsetP, 'recal': trainsetRecal, 'f1score': trainsetF1}
    myModel['testsetPerformance'] = {'tn': int(testsetCM[0][0]) ,'fp': int(testsetCM[0][1]),'fn': int(testsetCM[1][0]),'tp': int(testsetCM[1][1]), 'precision': testsetP, 'recal': testsetRecal, 'f1score': testsetF1}

    if(verbose):
        print("Matrice de confusion total : ", datasetCM)
        print("Matrice de confusion training set : ", trainsetCM)
        print("Matrice de confusion test set : ", testsetCM)
    
    return myModel
    
