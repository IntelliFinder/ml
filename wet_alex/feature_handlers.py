

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC

import pandas as pd
import numpy as np

import copy

from utils import customPipeline, CustomFeatureHandler

"""
Put all the operators on the features here
Implement in a pipeline as in the 'data_preparation.py' file
"""

class Template(CustomFeatureHandler):
    """DESCRIPTION"""

    def __init__(self):
        super().__init__()

    def transform(self, df : pd.DataFrame()):

        return df

"""=========================================================="""

class SyndromClassHandler(CustomFeatureHandler):
    """SyndromClassHandler"""

    def __init__(self):
        super().__init__()

    def transform(self, df : pd.DataFrame()):

        df.SyndromeClass = df.SyndromeClass.fillna(5)

        return df

class DisciplineScoreHandler(CustomFeatureHandler):
    """Handle the discipline score"""

    def __init__(self, mean_discipline):
        super().__init__()
        self.mean_discipline = mean_discipline

    def transform(self, df : pd.DataFrame()):

        # any value above 10 will be divided by 10


        outlier_discipline_score = df.DisciplineScore > 10

        df.loc[outlier_discipline_score, "DisciplineScore"] = df.loc[outlier_discipline_score, "DisciplineScore"] / 10

        # fill others with mean
        # replace the NA values with mean
        df.DisciplineScore = df.DisciplineScore.fillna(self.mean_discipline)

        return df


class SocialActivitiesHandler(CustomFeatureHandler):
    """Deal with social activities"""

    def __init__(self, mean_social_activity_time):
        super().__init__()
        self.mean_social_activity_time = mean_social_activity_time

    def transform(self, df : pd.DataFrame()):

        # there is linear dependency with 'AvgHouseholdExpenseOnPresents' parameter
        # so if such parameter exists, we use it to fill the missing data

        # find all the values where it is nan
        no_discipline_score_mask = df.DisciplineScore.isna()

        # check where it is not nan at AvgHouseholdExpenseOnPresents
        expenses_mask = ~ df.AvgHouseholdExpenseOnPresents.isna()

        # merge 2 masks
        discipline_refill_series = no_discipline_score_mask & expenses_mask

        df.loc[discipline_refill_series, "DisciplineScore"] = \
            df.loc[discipline_refill_series, "AvgHouseholdExpenseOnPresents"] / 10

        # all the other missing data will receive the mean value, since no other dependency was found.

        # replace the NA values with mean
        df.TimeOnSocialActivities = df.TimeOnSocialActivities.fillna(self.mean_social_activity_time)



        return df

class RemoveNotSRSColumns(CustomFeatureHandler):
    """Remove all the features that were decided that don't stay"""

    def __init__(self, srs_features_to_stay, more_features_to_stay):
        super().__init__()

        self.features_to_stay = srs_features_to_stay + more_features_to_stay

    def transform(self, df : pd.DataFrame()):

        for column in df.columns:
            if column not in self.features_to_stay:
                df = df.drop(columns = column)


        return df


class BloodTypeHandler(CustomFeatureHandler):
    """Add blood type to categories"""

    def __init__(self):
        super().__init__()

    def transform(self, df : pd.DataFrame()):
        df.BloodType = df.BloodType.astype('category')
        df.BloodType = df.BloodType.cat.add_categories("Unknown")
        df.BloodType = df.BloodType.fillna("Unknown")

        df["BloodTypeInt"] = df.BloodType.cat.rename_categories\
            (range(df.BloodType.nunique())).astype(int)
        df = df.drop(columns="BloodType")

        return df


class SexHandler(CustomFeatureHandler):
    """Deal with Sex categories"""

    def __init__(self):
        super().__init__()

    def transform(self, df : pd.DataFrame()):
        df.Sex = df.Sex.astype('category')
        df.Sex = df.Sex.cat.add_categories("No Gender")
        df.Sex = df.Sex.fillna("No Gender")

        df["SexInt"] = df.Sex.cat.rename_categories(range(df.Sex.nunique())).astype(int)
        df = df.drop(columns="Sex")

        return df

class AgeHandler(CustomFeatureHandler):
    """Deal with age class"""

    def __init__(self):
        super().__init__()

    def transform(self, df : pd.DataFrame()):
        df['AgeGroup'] = df.AgeGroup.fillna(np.mean(df.AgeGroup))

        return df


class SportsHandler(CustomFeatureHandler):
    """Deal with AvgMinSportsPerDay class"""

    def __init__(self):
        super().__init__()

    def transform(self, df : pd.DataFrame()):
        df['AvgMinSportsPerDay'] = df.AvgMinSportsPerDay.fillna(np.mean(df.AvgMinSportsPerDay))

        return df

class PresentsHandler(CustomFeatureHandler):
    """Deal with AvgHouseholdExpenseOnPresents class"""

    def __init__(self):
        super().__init__()

    def transform(self, df : pd.DataFrame()):
        df['AvgHouseholdExpenseOnPresents'] = df.AvgHouseholdExpenseOnPresents.fillna(np.mean(df.AvgHouseholdExpenseOnPresents))

        return df

class HappyHandler(CustomFeatureHandler):
    """Deal with AvgHouseholdExpenseOnPresents class"""

    def __init__(self):
        super().__init__()

    def transform(self, df : pd.DataFrame()):
        df['HappinessScore'] = df.HappinessScore.fillna(np.mean(df.HappinessScore))

        return df
class StepsHandler(CustomFeatureHandler):
    """Deal with steps class, normalize"""

    def __init__(self):
        super().__init__()

    def transform(self, df : pd.DataFrame()):
        std = StandardScaler()
        df['StepsPerYear'] = std.fit_transform(np.array(df.StepsPerYear.fillna(np.mean(df.StepsPerYear))).reshape(-1, 1))

        return df



class CousinsHandler(CustomFeatureHandler):
    """Deal with nr cousins class, normalize"""

    def __init__(self):
        super().__init__()

    def transform(self, df : pd.DataFrame()):
        std = StandardScaler()
        df['NrCousins'] = std.fit_transform(np.array(df.NrCousins.fillna(np.mean(df.NrCousins))).reshape(-1, 1))      

        return df


class SelfDeclaration_to_Categories(CustomFeatureHandler):
    """Parse each declaration into separate declarations"""

    def __init__(self):
        super().__init__()

    def transform(self, df : pd.DataFrame()):

        unique_self_declaracion_classes = []

        # string parser
        def parse_syndromes(syndromes_str: str) -> list:
            syndromes_list = []
            syndromes_str = syndromes_str.split(sep=";")

            for syndrome in syndromes_str:
                syndrome = syndrome.strip()
                syndromes_list.append(syndrome)

            return syndromes_list

        # create list of all syndromes
        for syndromes_list in df.SelfDeclarationOfIllnessForm.unique():
            if not isinstance(syndromes_list, str):
                continue

            syndromes = parse_syndromes(syndromes_list)
            for syndrome in syndromes:
                if syndrome not in unique_self_declaracion_classes:
                    unique_self_declaracion_classes.append(syndrome)

        # for each syndromes list, will return the dict with binary encoding for each syndrome
        def add_syndrom_categories(virus_df, syndromes_classes) -> pd.DataFrame:
            # parse each sample of syndromes_df
            # and set True for specific category

            syndromes_dict = {}

            for syndrom in syndromes_classes:
                syndromes_dict[syndrom] = np.zeros(virus_df.shape[0])

            for idx, sample in enumerate(virus_df.SelfDeclarationOfIllnessForm):
                if not isinstance(sample, str):
                    continue
                syndromes_list = parse_syndromes(sample)

                for syndrome in syndromes_list:
                    syndromes_dict[syndrome][idx] = 1

            return syndromes_dict

        # dict for each symptom, list of booleans
        syndromes_dict = add_syndrom_categories(df, unique_self_declaracion_classes)

        # change each to category in the dataset
        for key in syndromes_dict.keys():
            df[key] = syndromes_dict[key]
            df[key] = df[key].astype('category')

            # create the binary equivalent
            df[f"{key}Int"] = df[key].cat.rename_categories(range(df[key].nunique())).astype(int)
            df = df.drop(columns=key)

        df.drop(columns=['SelfDeclarationOfIllnessForm'])

        return df

class Drop_Irrelevant(CustomFeatureHandler):
    """Drop irrelevant features"""

    def __init__(self):
        super().__init__()

    def transform(self, df : pd.DataFrame()):
        df = df.drop(columns=['PatientID', 'Address']) # meaningless to disease diagnosis
        return df


class Modify_Results_Code(CustomFeatureHandler):
    """one-hot encode the test results"""

    def __init__(self):
        super().__init__()

    def transform(self, df: pd.DataFrame()):
        """Replace the test results with the new dataframe"""

        if 'TestResultsCode' in df.columns:

            # test result classified by labels
            li = df.TestResultsCode.tolist()
            labels = [item.split('_') for item in li]
            for item in labels:
                if len(item) == 4:
                    add = item[0] + item[1]
                    item = item.insert(0, add)
            for item in labels:
                if 'not' in item:
                    item.remove('not')
                if 'detected' in item:
                    item.remove('detected')

            disease = [la[0] for la in labels]
            spread = [la[1] for la in labels]
            risk = [la[2] for la in labels]

            # mapping dict
            # mapping = {}
            # for i,x in enumerate(set(disease)):
            #     mapping[x] = i

            # to be sure we're consistent. is also used in automatic classificaion
            disease_mapping = {'flue': 0, 'covid': 1, 'cmv': 2, 'cold': 3, 'measles': 4, 'notdetected': 5}

            disease_indexed = [disease_mapping[disease_name] for disease_name in disease]


            # disease_encode = pd.Series(disease).str.get_dummies()
            spread_encode = pd.Series(spread).str.get_dummies()
            risk_encode = pd.Series(risk).str.get_dummies()

            # disease_encode = pd.DataFrame(disease_encode)
            disease_encode = pd.DataFrame({'Disease' : disease_indexed})
            spread_encode = pd.DataFrame(spread_encode)
            risk_encode = pd.DataFrame(risk_encode)


            spread_encode = spread_encode.drop(['NotSpreader'], axis=1)
            risk_encode = risk_encode.drop(['NotatRisk'], axis=1)

            frames = [df, disease_encode, spread_encode, risk_encode]
            df = pd.concat(frames, axis=1)

            # drop the original label
            df = df.drop(columns='TestResultsCode')

        return df

class BMI_handler(CustomFeatureHandler):
    """Apply transformation on the BMI parameters"""

    def __init__(self, max_threshold: int, mean_bmi : int):
        super().__init__()
        self.max_threshold = max_threshold
        self.mean_bmi = mean_bmi

    def transform(self, df : pd.DataFrame()):

        # replace the outliers with mean
        outlier_bmi_mask = df.BMI > self.max_threshold

        df.loc[outlier_bmi_mask, "BMI"] = self.mean_bmi

        # replace the NA values with mean
        df.BMI = df.BMI.fillna(self.mean_bmi)

        return df


class PCR_results_handler(CustomFeatureHandler):
    """From the analysis, the PCR results 3, 12, and 16 can be removed"""
    def __init__(self, scaler_obj, pca_obj):
        super().__init__()
        self.scaler_obj = scaler_obj
        self.pca_obj = pca_obj

    def transform(self, df : pd.DataFrame()):

        pcr_results = range(17)
        pcr_fields = ["pcrResult" + str(i) for i in pcr_results]

        df_pcr = copy.deepcopy(df)

        # create new dataframe only with PCR results
        # fill NA values with median
        for column in df_pcr.columns:
            if column not in pcr_fields:
                df_pcr = df_pcr.drop(column, axis=1)
            else:
                df_pcr[column] = df_pcr[column].fillna(np.nanmedian(df_pcr[column]))

        # apply the Scaler on the whole dataframe
        df_pcr_scaled = self.scaler_obj.transform(df_pcr)

        # apply the PCA on the whole dataframe
        df_pcr_pca = self.pca_obj.transform(df_pcr_scaled)

        # Fit back the columns into the original dataset with field names PC1, PC2 ... etc
        for pca_component in range(df_pcr_pca.shape[1]):
            df[f"PCA_{pca_component}"] = df_pcr_pca[:, pca_component]


        # drop all the old pcr results from the original dataframe
        for column in df.columns:
            if column in pcr_fields:
                df = df.drop(column, axis=1)
        
        return df




class DropNA(CustomFeatureHandler):
    """Drop all columns, which have any N/A value"""

    def __init__(self):
        super().__init__()

    def transform(self, df : pd.DataFrame()):
        return df.dropna(axis=1)
