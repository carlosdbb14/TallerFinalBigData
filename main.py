from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, to_timestamp


spark = SparkSession.builder.appName('lab_SparkApi').getOrCreate()

sc = spark.sparkContext

df = spark.read.csv('input/HI-Small_Trans.csv',header = True, inferSchema=True)
df = df.withColumn('Timestamp', to_timestamp(df['Timestamp'],'yyyy/MM/dd HH:mm'))

df.show()

df.printSchema()