import pandas as pd                 # data analysis and manipulation tool
import numpy as np                  # Numerical computing tools
import seaborn as sns               # visualization library
import matplotlib.pyplot as plt     # another visualization library


from sklearn import decomposition
from sklearn.preprocessing import StandardScaler



def learn_pcr_transform(df : pd.DataFrame, n_components : int):

    sc = StandardScaler()
    pca = decomposition.PCA(n_components=n_components)

    # make dataframe only with pcr results
    pcr_results_to_remove = range(17)
    fields_to_remain = ["pcrResult" + str(i) for i in pcr_results_to_remove]

    for column in df.columns:
        if column not in fields_to_remain:
            df = df.drop(column, axis=1)

    # drop rows where at least 1 data is NA
    df = df.dropna(axis=0)

    # learn scaler on the train dataset
    df_scaled = sc.fit_transform(df)

    # find the PCA components
    pca.fit_transform(df_scaled)


    return sc, pca


def get_pca_columns(df : pd.DataFrame):

    pca = decomposition.PCA(n_components=5)
    pca.fit_transform(df)  # plug in scaled values ( with outliers )
    V = pca.components_

    cov = np.zeros((5, 4))
    for i in range(5):
        sort = np.sort(np.absolute(V[i]))
        for j in range(4):
            cov[i][j] = sort[15 - j]
    cov_idx = np.zeros((5, 4))

    for i in range(5):
        where = [(idx + 1) for idx, item in enumerate(V[i]) if
                 np.absolute(item) >= cov[i][3]]  # indices of 3 maximal coefficients
    for j in range(4):
        cov_idx[i][j] = where[j]

    keep = [int(cov_idx[i][j]) for i in range(3) for j in range(4)]
    keep_cols = ['pcrResult{}'.format(itr) for itr in keep]

    return keep_cols



def get_srs_features(df):

    """original code that was run on the training dataset to discover the best features to keep according to SRS"""

    #test result classified by labels
    li = df.TestResultsCode.tolist()
    labels =  [ item.split('_') for item in li ]
    for item in labels:
        if len(item)==4:
            add =  item[0]+item[1]
            item = item.insert( 0, add  )
    for item in labels:
        if 'not' in item:
            item.remove('not')
        if 'detected' in item:
            item.remove('detected')


    #one-hot encode the test results
    disease = [ la[0] for la in labels ]
    spread = [  la[1] for la in labels  ]
    risk = [  la[2] for la in labels  ]

    disease_encode = pd.Series( disease  ).str.get_dummies()
    spread_encode = pd.Series( spread  ).str.get_dummies()
    risk_encode = pd.Series( risk  ).str.get_dummies()

    disease_encode = pd.DataFrame( disease_encode )
    spread_encode = pd.DataFrame( spread_encode )
    risk_encode = pd.DataFrame( risk_encode)

    #interate one hot encoding of test results back to df
    df=df.drop(['PatientID', 'Address', 'CurrentLocation'],axis=1)
    df2 = df
    df2 = df2.drop(columns = 'TestResultsCode')

    results = pd.concat( [risk_encode, spread_encode, disease_encode], axis=1 )
    results = results.drop(['NotSpreader', 'NotatRisk'], axis=1)

    from sklearn.model_selection import train_test_split

    X_train,  X_val, y_train, y_val = train_test_split( df2, results, test_size=0.33, random_state=33 ) #tr is test results numerically coded
    X_val, X_test, y_val, y_test = train_test_split( X_val, y_val , test_size=0.4, random_state=33)

    #REMOVED LOCATION FROM FEATURES

    # choosing from those features
    cols =['AgeGroup','AvgHouseholdExpenseOnPresents','AvgHouseholdExpenseOnSocialGames',
            'AvgHouseholdExpenseParkingTicketsPerYear','AvgMinSportsPerDay','AvgTimeOnSocialMedia','AvgTimeOnStuding','BMI',
            'DisciplineScore','HappinessScore','Job','NrCousins','StepsPerYear','SyndromeClass','TimeOnSocialActivities']

    X_train_sfs = X_train[cols]
    X_train_sfs = X_train_sfs.fillna(X_train_sfs.mean())

    from sklearn.neighbors import KNeighborsClassifier
    from mlxtend.feature_selection import SequentialFeatureSelector as SFS #import from mlxtend library
    knn = KNeighborsClassifier(n_neighbors=2) # ml_algo used = knn
    sfs = SFS(knn,
               k_features=10,
               forward=True, # if forward = True then SFS otherwise SBS
               floating=False,
               verbose=2,
               scoring='accuracy'
               )


    #after applying sfs fit the data:
    sfs.fit(X_train_sfs, y_train)

    return sfs.k_feature_names_