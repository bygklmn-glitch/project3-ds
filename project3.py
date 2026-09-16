import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Create dataset
data = {
    "Customer_ID": range(1, 11),
    "Annual_Income": [30000,60000,40000,25000,50000,70000,35000,80000,45000,32000],
    "Spending_Score": [60,40,70,80,50,30,65,20,55,75]
}
df = pd.DataFrame(data)

X = df[['Annual_Income','Spending_Score']]
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Graph: Cluster Plot
plt.scatter(df['Annual_Income'], df['Spending_Score'], c=df['Cluster'], cmap='viridis')
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")
plt.show()
