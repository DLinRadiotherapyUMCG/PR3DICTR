# -*- coding: utf-8 -*-


from src.log_reg.generic_log_reg import lr_load_data, lr_prepare_data, lr_fit_model, lr_print_model_coefficients, lr_save_model, train_lr
from sklearn.metrics import roc_auc_score

import numpy as np

def load_CITOR_features(config,relevant_columns, train_data, val_data, toxicity):
    label = config['CITOR'][toxicity]['label']
    X_train = train_data[relevant_columns][train_data[label] != -1]
    y_train = train_data[label][train_data[label] != -1]
    X_val = val_data[relevant_columns][val_data[label] != -1]
    y_val = val_data[label][val_data[label] != -1]    
    return X_train, y_train, X_val, y_val

def CITOR_refit(config, toxicity = 'sticky_saliva_late'):
    
    train_data, val_data = lr_load_data(config)
    
    if len(config['CITOR'][toxicity]['models'].keys()) >1:
        feature_dict = {'model1':{},'model2':{}}
        model_names = ['model1','model2']
    else:
        feature_dict = {'model1':{}}
        model_names = ['model1']     

    for model_name in model_names:
       # select the columns per submodel
        relevant_columns = np.array(config['CITOR'][toxicity]['models'][model_name]['features'])
        # train the submodel
        X_train, y_train, X_val, y_val = load_CITOR_features(config,relevant_columns, train_data, val_data, toxicity)
        model, auc = lr_fit_model(config, X_train, y_train, X_val, y_val)

        # save the results from the submodel
        feature_dict[model_name]['features'] = relevant_columns
        feature_dict[model_name]['coef'] = model.coef_[0]
        feature_dict[model_name]['intercept'] = model.intercept_[0]
    
    if len(feature_dict.keys()) > 1:
        model1 = list(feature_dict.keys())[0]
        model2 = list(feature_dict.keys())[1]
        features = []
        coef = []
        features.append('intercept')
        coef.append(.5*(feature_dict[model1]['intercept']+feature_dict[model2]['intercept']))
        
        features1 = feature_dict[model1]['features']
        features2 = feature_dict[model2]['features']
        
        overlap =  list(set(features1) & set(features2))
        remainder_m1 = list(set(features1) - set(features2))
        remainder_m2 = list(set(features2) - set(features1))

        for i in range(len(overlap)):
            features.append(overlap[i])
            coef.append(.5*(feature_dict[model1]['coef'][np.where(features1== overlap[i])]+feature_dict[model2]['coef'][np.where(features2 == overlap[i])])[0])
        for i in range(len(remainder_m1)):
            features.append(remainder_m1[i])
            coef.append(.5*(feature_dict[model1]['coef'][np.where(features1 == remainder_m1[i])])[0])         
        for i in range(len(remainder_m2)):
            features.append(remainder_m2[i])
            coef.append(.5*(feature_dict[model2]['coef'][np.where(features2== remainder_m2[i])])[0])         
   
    else:
        features =['intercept'] +  list(relevant_columns)
        coef =[model.intercept_[0]]+ list(model.coef_[0]) 
    
    label = config['CITOR'][toxicity]['label']
    intercept = coef[0]
    X = train_data[features[1:]][train_data[label] !=-1].to_numpy()
    y = train_data[label][train_data[label] != -1]
    
    pred = 1/(1+np.exp(-(intercept + np.dot(X,coef[1:]))))
    
    auc = roc_auc_score(y, pred)
    print('auc:',auc)
    
    return features, coef