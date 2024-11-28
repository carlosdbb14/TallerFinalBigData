from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, to_date ,col, when


spark = SparkSession.builder.appName('lab_SparkApi').getOrCreate()

sc = spark.sparkContext

df = spark.read.csv('input/HI-Small_Trans.csv',header = True, inferSchema=True)


df.printSchema()

#limpiar y arreglar datos

#Paso la fecha a formato Fecha sin hora
df = df.withColumn('Timestamp', to_timestamp(df['Timestamp'],'yyyy/MM/dd HH:mm'))
df = df.withColumn('Timestamp', to_date(col('Timestamp')))

#agrego una columna que me diga si el banco de entrada y salida son el mismo banco

df = df.withColumn(
    'SameBank', 
    when(col('From Bank') == col('To Bank'),1).otherwise(0)
)


#crear una columna sobre si es dentro del mismo banco



## hacer analisis exploratorio de datos, mirar cuantos fraudes hay, con que monedas, cuantos datos hay de cada moneda, cuanto dinero se mueve, cuantos no fraudulentos hay, relaciones entre variables, etc, etc, eso se puede dejar en planos con los datos resumidos (ojala, los planos unidos)

df = df.withColumn('iguales',col('Amount Received') == col('Amount Paid'))

df.show()

df.groupBy('iguales').count().show()
#modelos para compararlos

#hacer un random forest

#hacer un xgboost

#comparar los dos


