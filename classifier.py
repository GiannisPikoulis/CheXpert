from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import TruncatedSVD
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import argparse
import time
import os
import sys
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn import set_config
#from imblearn.over_sampling import RandomOverSampler

SEED = 123
np.random.seed(SEED)

class CheXpertDataset(data.Dataset):

    def __init__(self, label_strategy, version='small', mode='train', path='/gpu-data2/jpik', transform=None):

        # Change the path accordingly
        self.path = path
        self.transform = transform
        self.mode = mode
        self.strategy = label_strategy

        self.conditions = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
                           "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
                           "Fracture", "Support Devices"]

        self.attributes = ["Sex", "Age", "Frontal/Lateral", "AP/PA"]

        self.df = pd.read_csv(os.path.join(self.path, 'CheXpert-v1.0-{}/{}.csv'.format(version, mode)))

        # Replace NaN condition values with zeros
        self.df = self.df.fillna(value=dict.fromkeys(self.conditions, 0))

        # Uncertain label replacement
        if self.mode == 'train' and self.strategy == 'U-Zeros':
            self.df = self.df.replace(dict.fromkeys(self.conditions, -1), 0)
        elif self.mode == 'train' and self.strategy == 'U-Ones':
            self.df = self.df.replace(dict.fromkeys(self.conditions, -1), 1)
        
        self.targets = self.df[self.conditions]


    def __getitem__(self, index):

        conditions = self.df.iloc[index][self.conditions]

        fname = os.path.join(self.path, self.df.iloc[index]['Path'])

        img = Image.open(os.path.join(self.path, fname)).convert("RGB")

        if self.transform is None:
            process_img = img
        else:
            process_img = self.transform(img)

        return process_img, torch.tensor(conditions).float()

    def __len__(self):
        return len(self.df)

def roc_auc(output, target):
    # print(np.sum(target.cpu().detach().numpy(),axis=1),np.sum(target.cpu().detach().numpy(),axis=0))
    # print(output.size())
    return sklearn.metrics.roc_auc_score(target, output, average=None)

def main(args):

    X_train = np.load(args.train)    
    X_valid = np.load(args.valid)
    print(X_train.shape)
    print(X_valid.shape)
    train_dataset = CheXpertDataset(mode="train", version='small', label_strategy=args.strategy,
                                    transform=None)

    y_train = np.array(train_dataset.targets)
    
    valid_dataset = CheXpertDataset(mode="valid", version='small', label_strategy=args.strategy,
                                    transform=None)
    
    y_valid = np.array(valid_dataset.targets) 
    print(y_train.shape)
    print(y_valid.shape)   

    if args.classifier == 'svm':
        tsvd = TruncatedSVD(random_state=SEED)
        clf = OneVsRestClassifier(SVC(random_state=SEED))

        kernel = ['rbf']
        gamma = ['auto']
        degree = np.arange(3, 4)
        n_components = [400]
        tsvd_algorithm = ['randomized']
        pipe = Pipeline(steps = [('tsvd', tsvd), ('svm', clf)])
        print(pipe)

        estimator = GridSearchCV(pipe, [{'tsvd__n_components': n_components, 'tsvd__algorithm': tsvd_algorithm, 
                                        'svm__estimator__kernel': kernel, 'svm__estimator__gamma': gamma, 'svm__estimator__degree': degree}, 
                                        {'tsvd': ['passthrough'], 
                                        'svm__estimator__kernel': kernel, 'svm__estimator__gamma': gamma, 'svm__estimator__degree': degree}], 
                                cv = args.cv, scoring = 'roc_auc', n_jobs = args.workers, verbose = 2)

        start_time = time.time()
        estimator.fit(X_train, y_train)
        print("Total time for GridSearchCV: {:.3f} seconds".format(time.time() - start_time))
        print("Mean fit time: {:.3f} seconds".format(np.mean(estimator.cv_results_['mean_fit_time'])))
        print("Mean score time: {:.3f} seconds".format(np.mean(estimator.cv_results_['mean_score_time'])))
        start_time = time.time()
        preds = estimator.best_estimator_.predict(X_valid)
        #print("Total time for inference on test set: {:.3f} seconds\n".format(time.time() - start_time))
        #print(estimator.best_estimator_, '\n')
        #print(estimator.best_params_)
        #print(classification_report(y_valid, preds, digits = 4,  target_names = train_dataset.conditions))
        conditions = train_dataset.conditions
        preds = preds[:, [x for x in range(14) if x != 12]]
        y_valid = y_valid[:, [x for x in range(14) if x != 12]]
        conditions = conditions[:12] + [conditions[13]]
        ra = roc_auc(preds, y_valid)
        print(np.mean(ra))
        for j in range(len(conditions)):
            print(conditions[j], ra[j])
        task_inds = [2, 5, 6, 8, 10]
        ra = ra[task_inds]
        print('Task ROC-AUC: ', np.mean(ra))    
    elif args.classifier == 'lr':
        tsvd = TruncatedSVD(random_state=SEED)
        clf = OneVsRestClassifier(LogisticRegression(random_state=SEED))
        solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'] 
        penalty = ['l2', 'l1', 'elasticnet']
        n_components = [200, 400, 600, 800] 
        tsvd_algorithm = ['randomized']
        pipe = Pipeline(steps = [('tsvd', tsvd), ('lr', clf)])
        print(pipe)
       # estimator = GridSearchCV(pipe, [{'tsvd__n_components': n_components, 'tsvd__algorithm': tsvd_algorithm,
#                                        'svm__estimator__kernel': kernel, 'svm__estimator__gamma': gamma, 'svm__estimator__degree': degree},
#                                        {'tsvd': ['passthrough'],
#                                        'svm__estimator__kernel': kernel, 'svm__estimator__gamma': gamma, 'svm__estimator__degree': degree}],
#                                cv = args.cv, scoring = 'roc_auc', n_jobs = args.workers, verbose = 2)
        estimator = GridSearchCV(pipe, [{'tsvd__n_components': n_components, 'tsvd__algorithm': tsvd_algorithm, 
                                         'lr__estimator__penalty': penalty, 'lr__estimator__solver': solver},
                                        {'tsvd': ['passthrough'], 
                                         'lr__estimator__penalty': penalty, 'lr__estimator__solver': solver}], 
                                 cv = args.cv, scoring = 'roc_auc', n_jobs = args.workers, verbose = 2)

        start_time = time.time()
        estimator.fit(X_train, y_train)
        print("Total time for GridSearchCV: {:.3f} seconds".format(time.time() - start_time))
        print("Mean fit time: {:.3f} seconds".format(np.mean(estimator.cv_results_['mean_fit_time'])))
        print("Mean score time: {:.3f} seconds".format(np.mean(estimator.cv_results_['mean_score_time'])))
        start_time = time.time()
        preds = estimator.best_estimator_.predict(X_valid)
        #print("Total time for inference on test set: {:.3f} seconds\n".format(time.time() - start_time))
        #print(estimator.best_estimator_, '\n')
        #print(estimator.best_params_)
        #print(classification_report(y_valid, preds, digits = 4,  target_names = train_dataset.conditions))
        conditions = train_dataset.conditions
        preds = preds[:, [x for x in range(14) if x != 12]]
        y_valid = y_valid[:, [x for x in range(14) if x != 12]]
        conditions = conditions[:12] + [conditions[13]]
        ra = roc_auc(preds, y_valid)
        print(np.mean(ra))
        for j in range(len(conditions)):
            print(conditions[j], ra[j])
        task_inds = [2, 5, 6, 8, 10]
        ra = ra[task_inds]
        print('Task ROC-AUC: ', np.mean(ra))
    else:
        raise NotImplementedError                            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classifier training and evaluation on the CheXpert dataset')
    parser.add_argument('--train', required=True, type=str, help='train features file path')
    parser.add_argument('--valid', required=True, type=str, help='validation features file path')
    parser.add_argument('--classifier', required=True, type=str, choices=["svm", "lr", "knn", 'mlp', "rf"], help='classifier to use')
    parser.add_argument('--strategy', type=str, default="U-Zeros", choices=["U-Zeros", "U-Ones"], help="Uncertain condition label replacement strategy (default: %(default)s)")
    parser.add_argument('--cv', default=None, type=int, help='cv splits (default: %(default)s)')
    parser.add_argument('--workers', default=1, type=int, help='number of workers (default: %(default)s)')
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    main(args)    
