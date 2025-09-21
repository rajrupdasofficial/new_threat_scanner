from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("DatasetExplorer").getOrCreate()

# Read the CSV with header enabled and comma delimiter
df = spark.read.option("header", "true").option("delimiter", ",").csv("dataset/vendors.csv")

# Print schema to verify data types (optional but helpful)
df.printSchema()

# Get and print column names
column_names = df.columns
print(column_names)
