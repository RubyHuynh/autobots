import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
import pandas as pd

# Define keywords for manual anomaly detection
manual_keywords = ["error", "fail", "exception", "critical"]

def load_logs(log_folder):
    log_texts = []
    log_paths = []
    for filename in os.listdir(log_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(log_folder, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                log_texts.extend(lines)
                log_paths.extend([file_path] * len(lines))
    return log_texts, log_paths

def vectorize_logs(log_texts):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(log_texts)
    return X, log_texts

def detect_anomalies(X, log_texts, log_paths):
    # Automatic anomaly detection using IsolationForest
    model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    model.fit(X)
    predictions = model.predict(X)
    anomaly_df = pd.DataFrame({'log': log_texts, 'path': log_paths, 'anomaly': predictions})
    
    # Manual anomaly detection using keywords
    manual_anomalies = anomaly_df[anomaly_df['log'].str.contains('|'.join(manual_keywords), case=False, na=False)]
    
    # Combine results
    anomalies = pd.concat([anomaly_df[anomaly_df['anomaly'] == -1], manual_anomalies]).drop_duplicates()
    normal_entries = anomaly_df[~anomaly_df.index.isin(anomalies.index)]
    
    # Sorting by file path
    anomalies.sort_values(by='path', inplace=True)
    normal_entries.sort_values(by='path', inplace=True)
    
    # Count and print statistics
    print(f"Normal entries: {len(normal_entries)}")
    print(f"Abnormal entries: {len(anomalies)}")
    
    # Print patterns
    print("\nNormal log patterns:")
    print(set(normal_entries['log']))
    print("\nAbnormal log patterns:")
    print(set(anomalies['log']))
    
    return anomalies

if __name__ == "__main__":
    log_folder = './scanlogs/'  # Replace with your log folder path
    log_texts, log_paths = load_logs(log_folder)
    X, log_texts = vectorize_logs(log_texts)
    anomalies = detect_anomalies(X, log_texts, log_paths)
    print("\nAnomalies detected with file paths:")
    print(anomalies[['log', 'path']])


''' sample result:
@RubyHuynh ➜ /workspaces/autobots (main) $ /home/codespace/.python/current/bin/python3 /workspaces/autobots/scanlog.py
/workspaces/autobots/scanlog.py:42: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame
See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  normal_entries.sort_values(by='path', inplace=True)
Normal entries: 14
Abnormal entries: 7

Normal log patterns:
{'info releasing user 0002\n', 'info creating new user 0004\n', 'info creating new user 0008', 'info creating new user 0002\n', 'info creating new user 0007\n', 'info releasing user 0003\n', 'info creating new user 0006\n', 'info creating new user 0005\n', 'info releasing 0xffdd45 ok\n', 'info releasing delivery service from user 0002 at 0xffdd45\n', 'info releasing user 0004\n', 'info creating new user 0003\n', 'info releasing user 0005\n'}

Abnormal log patterns:
{'info adding delivery service on user 0001 at 0xffdd45\n', 'error duplicated user\n', 'info releasing delivery service from user 0001 at 0xffdd46\n', 'error SIG11', 'error SIG11\n', 'error 0xffdd46 service not found\n', 'info adding delivery service on user 0002 at 0xffdd46\n'}

Anomalies detected with file paths:
                                                  log                 path
5                             error duplicated user\n  ./scanlogs/log1.txt
8                                       error SIG11\n  ./scanlogs/log1.txt
14  info adding delivery service on user 0001 at 0...  ./scanlogs/log2.txt
15  info adding delivery service on user 0002 at 0...  ./scanlogs/log2.txt
16  info releasing delivery service from user 0001...  ./scanlogs/log2.txt
17                 error 0xffdd46 service not found\n  ./scanlogs/log2.txt
20                                        error SIG11  ./scanlogs/log2.txt

'''