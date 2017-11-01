#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 15:12:05 2017

@author: Beth
"""
import pandas as pd
import numpy as np
import networkx as nx

from sklearn.feature_extraction import DictVectorizer
from sklearn import base
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


from collections import defaultdict
import pickle


def Predict():
    # ### Generate Data

    # generateFeatures(B)

    # ### Load Data
    with open('popularity.pickle', 'rb') as f:
        popularity = pickle.load(f)
#    print(popularity[:10])

    with open('features.pickle', 'rb') as f:
        features2 = pickle.load(f)
#    print(features2[:10])

    with open('aggdata.pickle', 'rb') as f:
        aggvals = pickle.load(f)
#    print(aggvals[:10])

    pop = pd.DataFrame.from_records(
            popularity, columns=["M1", "M2", "M3", "M4", "M5", "M6"])
    pop["Sum"] = pop["M1"] + pop["M2"] + pop["M3"] + pop["M4"] + pop["M5"]\
        + pop["M6"]

    feat = pd.DataFrame.from_records(features2,
                                     columns=["Creator", "Title", "Year",
                                              "Publishers", "Subjects"])
    aggregates = pd.DataFrame.from_records(
            aggvals, columns=["2011", "2012", "2013", "2014", "2015", "2016"])
    feat = pd.concat([feat, aggregates], axis=1)

#    G = nx.Graph()
#    nodes = {}
#    edges = {}
#    for i in range(len(feat)):
#        for j in feat["Publishers"][i]:
#            if j not in nodes:
#                nodes[j]=1
#            else:
#                nodes[j]+=1
#            for k in feat["Publishers"][i]:
#                if k<j:
#                    if((j,k) in edges):
#                        edges[(j,k)] += 1
#                    else:
#                        edges[(j,k)] = 1
#
#    print(nodes.items())
#    print(edges.items())
#    for a,b in nodes.items():
#        G.add_node(a, weight=b)
#    for (a,b),c in edges.items():
#        G.add_edge(a,b, weight=c)


    # Break off 2000 points from each of 2011-2016 as a final validation set
    features_train = []
    features_validate = []
    pop_train = []
    pop_validate = []
    for year in range(2011, 2017, 1):
        f_year_train, f_year_validate, pop_year_train, pop_year_validate\
            = train_test_split(feat.loc[feat["Year"] == year],
                               pop.loc[feat["Year"] == year]["Sum"],
                               test_size=2000,
                               random_state=20172017)
        features_train.append(f_year_train)
        features_validate.append(f_year_validate)
        pop_train.append(pop_year_train)
        pop_validate.append(pop_year_validate)
#    print(features_train[0][:10])


#    ### Tune the hyper-parameters
#    TuneParams("y", 0.01, 0.17, 0.03, features_train, pop_train, feat, pop)
#    # 0.01
#    TuneParams("s", 2,4.1,0.5, features_train, pop_train, feat, pop)
#    # 3, 22
#    TuneParams("p", 0.1, 3.6, 0.5, features_train, pop_train, feat, pop)
#    # 1, 20? Can't get above a negative value

#   ### Define the estimators
    author_est = Pipeline([
        ("Get Columns", ColumnSelectTransformer(["Creator"])),
        ("Model", CategoryEstimator())
        ])

    year_est = Pipeline([
        ("Get Columns", ColumnSelectTransformer(["Year"])),
        ("Model", Lasso(alpha=0.01))
        ])

    pub_est = Pipeline([
        ("Get Columns", ColumnSelectTransformer(["Publishers"])),
        ("Transform", DictEncoder()),
        ("Vectorize", DictVectorizer()),
        ("Model", LinNonLinCombiner(Lasso(alpha=1),
                                    RandomForestRegressor(max_depth=20)))
        ])
    title_est = Pipeline([
        ("Get Data", ColumnSelectTransformer(["Title"])),
        ("Get Tokens", TokenTransformer()),
        ("Estimate", CategoryEstimator())
        ])
    sub_est = Pipeline([
        ("Get Columns", ColumnSelectTransformer(["Subjects"])),
        ("Transform", DictEncoder()),
        ("Vectorize", DictVectorizer()),
        ("Model", LinNonLinCombiner(Lasso(alpha=3),
                                    RandomForestRegressor(max_depth=22)))
        ])
    auth_agg = Pipeline([
        ("Get Data", ColumnDeduceTransformer(["Creator"])),
        ("Estimator", AggregateEstimator())
        ])
    title_agg = Pipeline([
        ("Get Data", ColumnDeduceTransformer(["Title"])),
        ("Get Tokens", TokenTransformer()),
        ("Estimator", AggregateEstimator())
        ])


#   ### Tune the ensemble model
    for alph in np.arange(0.1, 4.2, 1):  #
        full_model = Pipeline([
            ("Get Features", FeatureUnion([
                # FeatureUnions use the same syntax as Pipelines
                ("Author Est", EstimatorTransformer(author_est)),
                ("Year Est", EstimatorTransformer(year_est)),
#                ("Publisher Est", EstimatorTransformer(pub_est)),
                ("Subject Est", EstimatorTransformer(sub_est)),
                ("Title Est", EstimatorTransformer(title_est)),
                ("Author Agg", EstimatorTransformer(auth_agg)),
                ("Title Agg", EstimatorTransformer(title_agg))
                ])),
            ("Model", LinNonLinCombiner(Lasso(alpha=alph),
                                        RandomForestRegressor(max_depth=20)))
        ])
        score = runTest(full_model, features_train, pop_train, feat,
                        pop, 10)
        print("Full Model score:", alph, score)

#   ### Define the ensemble model
    full_model = Pipeline([
            ("Get Features", FeatureUnion([
                # FeatureUnions use the same syntax as Pipelines
                ("Author Est", EstimatorTransformer(author_est)),
                ("Year Est", EstimatorTransformer(year_est)),
#                ("Publisher Est", EstimatorTransformer(pub_est)),
                ("Subject Est", EstimatorTransformer(sub_est)),
                ("Title Est", EstimatorTransformer(title_est)),
                ("Author Agg", EstimatorTransformer(auth_agg)),
                ("Title Agg", EstimatorTransformer(title_agg))
                ])),
            ("Model", LinNonLinCombiner(Lasso(alpha=1),
                                        RandomForestRegressor(max_depth=19)))
        ])


#   ### Cross-validation scores:

    # Author - no variation
    auth_val = runValidation(author_est, feat, pop["Sum"],
                             features_validate, pop_validate)
    print("Author score, validation:", auth_val)

    # Year - no variation (Lasso is deterministic?)
    year_val = runValidation(year_est, feat, pop["Sum"],
                             features_validate, pop_validate)
    print(" Year score, validation:", year_val)

    # Publisher - variation from Random Forest
    pub_vals = []
    for i in range(10):
        pub_vals.append(runValidation(pub_est, feat, pop["Sum"],
                                      features_validate, pop_validate))
    print("Publisher score, validation:", sum(pub_vals)/10)

    # Subject - variation from Random Forest
    sub_vals = []
    for i in range(10):
        sub_vals.append(runValidation(sub_est, feat, pop["Sum"],
                                      features_validate, pop_validate))
    print("Subject score, validation:", sum(sub_vals)/10)

    # Title - no variation
    title_val = runValidation(title_est, feat, pop["Sum"],
                              features_validate, pop_validate)
    print("Title score, validation:", title_val)

    # Ensemble model
    full_vals = []
    for i in range(10):
        full_vals.append(runValidation(full_model, feat, pop["Sum"],
                         features_validate, pop_validate))
    print("Full model score, validation", sum(full_vals)/10)


def runTest(pipe, feat, pop, all_feat, all_pop, numtries):

    preds = [[] for i in range(numtries)]
    results = [[] for i in range(numtries)]

#        no_auth = 0
#        no_auth_ok = 0
#        no_title = 0
#        no_title_ok = 0
#        total_bad = 0

    for y in range(2011, 2017, 1):
        print(y)
        pipe.fit(all_feat.loc[all_feat["Year"] < y],
                 all_pop["Sum"].loc[all_feat["Year"] < y])

        for trial in range(numtries):
            # Break up the data, selecting 9000 to evaluate
            features_test, features_val, pop_test, pop_val = \
                train_test_split(feat[y-2011], pop[y-2011], train_size=9000)
            full_pred = pipe.predict(feat[y-2011])
#            print(full_pred[:10])
            preds[trial].extend(full_pred)
            results[trial].extend(pop[y-2011])

#    #        print(full_pred[:10])
#    #        print(pop_test[:10])
#            print(len(full_pred))
#            ae = pipe.steps[0][1].transformer_list[0][1].estimator
#            auth_def = ae.steps[1][1].get_default()
#            auth_empty = ae.steps[1][1].predict([[""]])[0]
#            print(auth_def, auth_empty)
#            te = pipe.steps[0][1].transformer_list[4][1].estimator
#            title_def = te.steps[2][1].get_default()
#            title_empty = te.steps[2][1].predict([[""]])[0]
#            print(title_def, title_empty)
#            for i in range(len(full_pred)):
#                if abs(full_pred[i]-pop_test.iloc[i]) > 10:
#                    total_bad += 1
#
#                    aguess = ae.predict(features_test[i:i+1])[0]
#                    if aguess == auth_def or aguess == auth_empty:
#                        no_auth += 1
#                    tguess = te.predict(features_test[i:i+1])[0]
#                    if tguess == title_def or tguess == title_empty:
#                        no_title += 1
#                else:
#                    aguess = ae.predict(features_test[i:i+1])[0]
#                    if aguess == auth_def or aguess == auth_empty:
#                        no_auth_ok += 1
#                    tguess = te.predict(features_test[i:i+1])[0]
#                    if tguess == title_def or tguess == title_empty:
#                        no_title_ok += 1
#
#        print(total_bad, no_auth, no_title, no_auth_ok, no_title_ok)

    # ### Score all of the results
    scores = []
    for trial in range(numtries):
        scores.append(metrics.r2_score(results[trial], preds[trial]))
    return(np.mean(scores))


def runValidation(pipe, feat, pop, feat_val, pop_val):
    preds = []
    results = []
    for y in range(2011, 2017, 1):
        pipe.fit(feat.loc[feat["Year"] < y], pop.loc[feat["Year"] < y])
        full_pred = pipe.predict(feat_val[y-2011])
        preds.extend(full_pred)
        results.extend(pop_val[y-2011])
    score = metrics.r2_score(results, preds)
    return(score)


class EstimatorTransformer(base.BaseEstimator, base.TransformerMixin):
    '''
    Turns an estimator into a transformer:
        returns a column of predictions
    '''
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def transform(self, X):
        pred = self.estimator.predict(X)
        return([[x] for x in pred])


class LinNonLinCombiner(base.BaseEstimator, base.RegressorMixin):
    '''
    Combines a linear model with a non-linear model for the residuals
    '''
    def __init__(self, lin_est, nonlin_est):
        self.lin = lin_est
        self.nonlin = nonlin_est

    def fit(self, X, y):
        self.lin.fit(X, y)
        self.nonlin.fit(X, y-self.lin.predict(X))

    def predict(self, X):
        return(self.lin.predict(X) + self.nonlin.predict(X))


class DictEncoder(base.BaseEstimator, base.TransformerMixin):
    '''
    Converts a list of lists of lists into a list of dictionaries with counts.
    '''
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X will come in as a list of lists of lists.
        final = []
        for x in X:
            local = {}
            for j in x:
                for k in j:
                    local[k] = 1
            final.append(local)
        return(final)


class CategoryEstimator(base.BaseEstimator, base.RegressorMixin):
    '''
    Estimates based on the average of observations that match on the supplied
        features. Used for category data - author, title, etc.
    '''
    def __init__(self):
        self.avg = 0
        self.catDict = {}

    def fit(self, X, y):
        # print(X[:10], y[:10])
        sum_dict = defaultdict(int)
        count_dict = defaultdict(int)
        total_sum = 0
        total_count = 0
        for key, val in zip(X, y):
            # print(key, val)
            sum_dict[key[0]] += float(val)
            total_sum += float(val)
            count_dict[key[0]] += 1
            total_count += 1
        for key in sum_dict:
            self.catDict[key] = float(sum_dict[key])/count_dict[key]
        self.avg = total_sum/total_count

    def predict(self, X):
        predictions = []
        for x in X:
            if x[0] in self.catDict:
                predictions.append(self.catDict[x[0]])
            else:
                predictions.append(self.avg)
#        print(predictions)
        return(predictions)

    def get_default(self):
        return(self.avg)


class AggregateEstimator(base.BaseEstimator, base.RegressorMixin):

    def __init__(self):
        self.sumDict = defaultdict(int)

    def fit(self, X, y):
        # print(X[:10], y[:10])
        for key, val in zip(X, y):
            # print(key, val)
            self.sumDict[key[0]] += float(key[1])

    def predict(self, X):
        predictions = []
        for x in X:
            if x[0] in self.sumDict:
                predictions.append(self.sumDict[x[0]])
            else:
                predictions.append(0)
        return(predictions)

    def get_default(self):
        return(0)


class ColumnSelectTransformer(base.BaseEstimator, base.TransformerMixin):
    '''
    Selects a set of columns from X and returns them in the right format
        Good for setting up different pipelines.
    '''

    def __init__(self, col_names):
        self.col_names = col_names  # We will need these in transform()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # print(self.col_names)
        # for name in self.col_names:
        #     print(X[name])
        cols = [[X[name].iloc[i] for name in self.col_names]
                for i in range(len(X))]
#        print(cols)
        return(cols)


class ColumnDeduceTransformer(base.BaseEstimator, base.TransformerMixin):
    '''
    Selects the right year from the data it's fit on, then transforms data
        Good for setting up aggregate feature creators
    '''

    def __init__(self, col_names):
        self.col_names = col_names

    def fit(self, X, y=None):
        maxdate = max(X["Year"])
        self.col_names.append(str(maxdate + 1))
        return self

    def transform(self, X):
        cols = [[X[name].iloc[i] for name in self.col_names]
                for i in range(len(X))]
        return(cols)


class TokenTransformer(base.BaseEstimator, base.TransformerMixin):
    '''
    Specialized code for extracting series information from a book title.
        Takes in a column of titles, returns a column of just the tokens.
    '''
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        results = []
        for x in X:
            x[0] = extractToken(x[0])
            results.append(x)
        return(results)


def extractToken(title):
    '''
    Workhorse for token extraction, run by TokenTransformer.
    '''
    if ':' in title:
        secondhalf = title.split(':')[1]
    else:
        secondhalf = title

    flags = [", Book ", " book ", " Issue ", ", Part ", " Level ",
             "Volume ", " Books", " Tome", "-Tome", " No.", " Case ",
             " #", " no.", " Libro ", ", libro ", " Livre ", " tome "]
    found = False
    for f in flags:
        if f in secondhalf:
            found = True
            token = secondhalf.split(f)[0]
            break

    if(not found):
        if " Book " in secondhalf and len(secondhalf.split(" Book ")[1]) < 3:
            token = secondhalf.split(" Book ")[0]
            found = True
        if len(secondhalf) > 0 and secondhalf[-1] in "0123456789"\
           and secondhalf[-2] not in "0123456789":
            found = True
            token = secondhalf[:-1]

    if (not found):
        token = ""

    if len(token) > 0 and token[-1] == ',':
            token = token[:-1]
#    if(len(token)>0 and ":" not in  title):
#        print(token, title)
    return(token)


class AverageEstimator(base.BaseEstimator, base.RegressorMixin):

    def __init__(self):
        self.av = 0

    def fit(self, X, Y):
        totalsum = 0
        totalcount = 0
        for y in Y:
            totalsum += y
            totalcount += 1
        self.av = float(totalsum)/totalcount

    def predict(self, X):
        results = [[self.av] for x in X]
        return(results)


def generateFeatures():

    Books = pd.read_pickle("/Users/Beth/Python/EBooks_all")

#   ### Make a date field (with a datetime), from the year and month cols
    Books["Date"] = pd.to_datetime(pd.DataFrame(
            {"Year": Books["CheckoutYear"],
             "Month": Books["CheckoutMonth"],
             "Day": [1]*len(Books)}))
    Books = Books.fillna('')

    B = Books.groupby(["Creator", "Title"])
    ["Creator", "Title", "Date", "Checkouts", "Publisher", "Subjects"]

#   ### Generate Features, popularity (score to regress on)
    output = []
    features = []
    aggregates = []
    yearslist = [2011, 2012, 2013, 2014, 2015, 2016]
    for g in [b for b in B.groups][:]:
        data = B.get_group(g)
        data = data.sort_values(by="Date")
        checkouts = data.groupby(["Date"]).sum()
        dr = pd.date_range(checkouts.index[0], checkouts.index[-1], freq='MS')
        checkouts = checkouts.reindex(dr, fill_value=0)
        checkouts = checkouts[:6]
        clist = [checkouts["Checkouts"].iloc[i] if i < len(checkouts) else 0
                 for i in range(6)]
        output.append(clist)

        aggsums = [data.loc[data["Date"] < pd.datetime(year=y, month=1, day=1)]
                   ["Checkouts"].sum() for y in yearslist]
        aggregates.append(aggsums)

        year1 = data["Date"].iloc[0].year
        pub1 = [data["Publisher"].iloc[0]]
        sub1 = data["Subjects"].iloc[0].split(',')
        sub1 = [s.strip() for s in sub1]
        for i in range(1, len(data)):
            if data["Publisher"].iloc[i] not in pub1 \
                    and data["Publisher"].iloc[i] != '':
                # print(g)
                # print(data["Publisher"].iloc[i], pub1)
                pub1.append(data["Publisher"].iloc[i])
            for s in data["Subjects"].iloc[i].split(','):
                if s.strip() not in sub1 and s != '':
                    # print(g,s,sub1)
                    sub1.append(s.strip())
        features.append([g[0], g[1], year1, pub1, sub1])
        # print(checkouts["Checkouts"].iloc[0])

    print(output[:10])
    with open('popularity.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(output, f)

    print(features[:10])
    with open('features.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(features, f)

    print(aggregates[:10])
    with open('aggdata.pickle', 'wb') as f:
        pickle.dump(aggregates, f)


def TuneParams(name, low, high, step, features_train, pop_train, feat, pop):

    if name == "y":
        # Year: tune the parameter in Lasso
        for alph in np.arange(low, high, step):
            year_est = Pipeline([
                ("Get Columns", ColumnSelectTransformer(["Year"])),
                ("Model", Lasso(alpha=alph))
            ])
            score = runTest(year_est, features_train, pop_train, feat, pop, 10)
            print("year score:", alph, score)

    elif name == "p":
        #  Publisher(s): Tune linear and non-linear estimators
        for alph in np.arange(low, high, step):  # 1, 20 works well
            pub_est = Pipeline([
                ("Get Columns", ColumnSelectTransformer(["Publishers"])),
                ("Transform", DictEncoder()),
                ("Vectorize", DictVectorizer()),
                ("Model", LinNonLinCombiner(
                        Lasso(alpha=alph), RandomForestRegressor(max_depth=20)))
            ])
            score = runTest(pub_est, features_train, pop_train, feat, pop, 10)
            print("publisher score:", alph, score)

    elif name == "s":
        # Subjects: Tune linear and non-linear estimators
        for alph in np.arange(low, high, step):  # 3, 22 works well
            sub_est = Pipeline([
                ("Get Columns", ColumnSelectTransformer(["Subjects"])),
                ("Transform", DictEncoder()),
                ("Vectorize", DictVectorizer()),
                ("Model", LinNonLinCombiner(
                        Lasso(alpha=alph), RandomForestRegressor(max_depth=22)))
                ])
            score = runTest(sub_est, features_train, pop_train, feat, pop, 10)
            print("subject score:", alph, score)


Predict()
