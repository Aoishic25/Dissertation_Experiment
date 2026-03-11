import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
import os

# Path to STRICTLY VALIDATED matrix
file_path = r'C:\Users\Janshrut\Desktop\Sem2\master_permission_matrix_FINAL.csv'

def generate_iterative_performance_table():
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # Load validated dataset
    df = pd.read_csv(file_path)
    X = df.drop(columns=['applications', 'label'])
    y = df['label'].values
    
    # Calculate rankings to identify "Best Features" 
    chi_selector = SelectKBest(chi2, k='all').fit(X, y)
    chi_ranked = [f for _, f in sorted(zip(chi_selector.scores_, X.columns), reverse=True)]
    pearson_ranked = X.corrwith(pd.Series(y)).abs().sort_values(ascending=False).index.tolist()

    # Steps of 10 to match your graph's X-axis
    ks = list(range(10, len(X.columns) + 1, 10))
    if len(X.columns) not in ks: 
        ks.append(len(X.columns))

    results_list = []

    print("Running iterative evaluation to find best features...")
    for k in ks:
        row = {'No_of_Permissions': k}
        for algo_name, ranked_list in [("Chi2", chi_ranked), ("Pearson", pearson_ranked)]:
            # This line selects only the "k" best features for this step 
            X_subset = X[ranked_list[:k]]
            
            X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.30, random_state=42)
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
            dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
            
            row[f'{algo_name}_RF'] = round(accuracy_score(y_test, rf.predict(X_test)), 4)
            row[f'{algo_name}_DT'] = round(accuracy_score(y_test, dt.predict(X_test)), 4)
            
        results_list.append(row)

    # Create the DataFrame
    df_results = pd.DataFrame(results_list)
    
    # FIND THE BEST FEATURES (Peak Accuracy)
    print("\n" + "="*60)
    print("   PEAK PERFORMANCE SUMMARY   ")
    print("="*60)
    for col in ['Chi2_RF', 'Chi2_DT', 'Pearson_RF', 'Pearson_DT']:
        best_idx = df_results[col].idxmax()
        best_k = df_results.loc[best_idx, 'No_of_Permissions']
        best_acc = df_results.loc[best_idx, col]
        print(f"{col:10} | Peak Accuracy: {best_acc:.4f} at k={best_k}")
    
    print("\n--- FULL ITERATIVE TABLE ---")
    print(df_results.to_string(index=False))

if __name__ == "__main__":

    generate_iterative_performance_table()
