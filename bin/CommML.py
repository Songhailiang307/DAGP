import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from joblib import dump, load
import argparse
import json

def load_data(geno_file, phe_file):
    geno = np.loadtxt(geno_file, dtype=int)
    x = geno[:, 1:]
    ids = geno[:, 0].astype(str).tolist()

    phe_dict = {}
    with open(phe_file, 'r') as f:
        for line in f:
            tokens = line.split()
            phe_dict[tokens[0]] = float(tokens[1])

    y = [phe_dict[str(id)] for id in ids if str(id) in phe_dict]
    x = np.array([x[i] for i in range(len(ids)) if str(ids[i]) in phe_dict])

    return x, np.array(y), ids

def save_results(ids, y_true_dict, y_pred, out_file):
    with open(out_file, 'w') as f:
        for i, id in enumerate(ids):
            if str(id) in y_true_dict:
                f.write(f"{id}\t{y_true_dict[str(id)]}\t{y_pred[i]}\n")

def train_model(model, x_train, y_train, param_grid, model_file, param_file):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    dump(best_model, model_file)

    with open(param_file, 'w') as f:
        json.dump(grid_search.best_params_, f)

    return best_model

def predict_and_evaluate(model, x_val, val_ids, val_phe_dict, out_val_gebv_file):
    y_val_pred = model.predict(x_val)
    y_val_pred = y_val_pred.flatten() if len(y_val_pred.shape) > 1 else y_val_pred
    save_results(val_ids, val_phe_dict, y_val_pred, out_val_gebv_file)
    y_true = [val_phe_dict.get(str(id), np.nan) for id in val_ids]
    y_pred = [y_val_pred[i] for i, id in enumerate(val_ids) if str(id) in val_phe_dict]
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred have different number of output ({len(y_true)} != {len(y_pred)})")
    
    r2 = r2_score(y_true, y_pred)
    print(f"R2 Score: {r2}")
    return r2

def parse_args():
    parser = argparse.ArgumentParser(description="Genotype Prediction")
    parser.add_argument('--rel_geno_file', type=str, required=True, help="Path to rel_fina.dat")
    parser.add_argument('--rel_phe_file', type=str, required=True, help="Path to rel.txt")
    parser.add_argument('--val_geno_file', type=str, required=True, help="Path to val_fina.dat")
    parser.add_argument('--val_phe_file', type=str, required=True, help="Path to val.txt")
    parser.add_argument('--model', type=str, required=True, choices=['SVR', 'RF', 'KRR', 'XGB', 'GBR', 'ElasticNet'], help="Model to use")
    return parser.parse_args()

def main():
    args = parse_args()
    
    x_train, y_train, _ = load_data(args.rel_geno_file, args.rel_phe_file)
    x_val, _, val_ids = load_data(args.val_geno_file, args.val_phe_file)
    
    val_phe_dict = {}
    with open(args.val_phe_file, 'r') as f:
        for line in f:
            tokens = line.split()
            val_phe_dict[tokens[0]] = float(tokens[1])

    if args.model == 'SVR':
        model = SVR()
        param_grid = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto']
        }
    elif args.model == 'RF':
        model = RandomForestRegressor()
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif args.model == 'KRR':
        model = KernelRidge()
        param_grid = {
            'alpha': [0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': [0.1, 0.01, 0.001, 0.0001]
        }
    elif args.model == 'XGB':
        model = XGBRegressor()
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0]
        }
    elif args.model == 'GBR':
        model = GradientBoostingRegressor()
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0]
        }
    elif args.model == 'ElasticNet':
        model = ElasticNet()
        param_grid = {
            'alpha': [0.1, 1, 10, 100],
            'l1_ratio': [0.1, 0.5, 0.7, 1.0],
            'max_iter': [1000, 5000, 10000]
        }

    best_model = train_model(model, x_train, y_train, param_grid, f'{args.model.lower()}_model.joblib', f'{args.model.lower()}_best_params.json')
    r2 = predict_and_evaluate(best_model, x_val, val_ids, val_phe_dict, f'val_gebv_{args.model}')

    print(f"Best {args.model} parameters saved in {args.model.lower()}_best_params.json")

if __name__ == "__main__":
    main()