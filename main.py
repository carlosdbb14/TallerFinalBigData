from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, to_timestamp


spark = SparkSession.builder.appName('lab_SparkApi').getOrCreate()

sc = spark.sparkContext

df = spark.read.csv('input/HI-Small_Trans.csv',header = True, inferSchema=True)
df = df.withColumn('Timestamp', to_timestamp(df['Timestamp'],'yyyy/MM/dd HH:mm'))

df.show()

df.printSchema()


## hacer analisis exploratorio de datos, mirar cuantos fraudes hay, con que monedas, cuantos datos hay de cada moneda, cuanto dinero se mueve, cuantos no fraudulentos hay, relaciones entre variables, etc, etc, eso se puede dejar en planos con los datos resumidos (ojala, los planos unidos)


#modelos para compararlos

#hacer un random forest

#hacer un xgboost

#comparar los dos


