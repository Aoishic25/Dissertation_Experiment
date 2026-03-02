import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

#Path to the master matrix
file_path = r'C:\Users\Janshrut\Desktop\Sem2\master_permission_matrix_FINAL.csv'
output_csv=r'C:\Users\Janshrut\Desktop\Sem2\RF_vs_DT_Full_Features.csv'

def run_full_feature_comparison():
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # 1.Load Data
    df = pd.read_csv(file_path)
    X = df.drop(columns=['applications', 'label'])
    y = df['label'].values
    print(f"Total validated permissions being used: {len(X.columns)}")
    
    # 2.70:30 Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    #3.Initialize Models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }
    results = []

    #4. Train and Evaluate
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results.append({
            "Algorithm": name,
            "Accuracy":round(accuracy_score(y_test,y_pred),4),
            "Precision":round(precision_score(y_test,y_pred),4),
            "Recall":round(recall_score(y_test,y_pred),4),
            "F1-Score":round(f1_score(y_test,y_pred),4)
        })
        
    #5. Save and Display Table
    results_df=pd.DataFrame(results)
    results_df.to_csv(output_csv,index=False)
    
    print("\n" + "="*60)
    print("      FINAL COMPARISON TABLE (ALL VALIDATED FEATURES)")
    print("="*60)
    print(results_df.to_string(index=False))
    print("-" * 60)
    print(f"Table saved to: {output_csv}")

if __name__ == "__main__":
    run_full_feature_comparison()