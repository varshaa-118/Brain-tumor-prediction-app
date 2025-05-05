import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Connect to PostgreSQL ----
conn = psycopg2.connect(
    dbname="brain_tumor_log",
    user="postgres",
    password="123456",
    host="localhost",
    port="5432"
)

# ---- Fetch Data ----
query = "SELECT * FROM predictions ORDER BY timestamp DESC"
df = pd.read_sql_query(query, conn)
conn.close()

# ---- Show Data ----
print(df.head())  # show top rows

# ---- Plot: Count of Each Class ----
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='predicted_class', palette="pastel")
plt.title("Tumor Class Prediction Count")
plt.xlabel("Tumor Class")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
