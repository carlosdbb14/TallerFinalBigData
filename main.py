from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier

spark = SparkSession.builder.appName('lab_SparkApi').getOrCreate()

sc = spark.sparkContext

df = spark.read.csv('input/HI-Small_Trans.csv',header = True, inferSchema=True)


df.printSchema()

#limpiar y arreglar datos

#Paso la fecha a formato Fecha sin hora
df = df.withColumn('Timestamp', F.to_timestamp(df['Timestamp'],'yyyy/MM/dd HH:mm'))
df = df.withColumn('Timestamp', F.to_date(col('Timestamp')))

#agrego una columna que me diga si el banco de entrada y salida son el mismo banco y lo mismo con la cuenta para cuando son el mismo banco

df = df.withColumn(
    'SameBank', 
    F.when(col('From Bank') == col('To Bank'),1).otherwise(0)
)

df = df.withColumn(
    'SameAccount', 
    F.when(col('Account2') == col('Account4'),1).otherwise(0)
)




## hacer analisis exploratorio de datos, mirar cuantos fraudes hay, con que monedas, cuantos datos hay de cada moneda, cuanto dinero se mueve, cuantos no fraudulentos hay, relaciones entre variables, etc, etc, eso se puede dejar en planos con los datos resumidos (ojala, los planos unidos)

#los datos de el dinero enviado

mons = df.groupBy("Payment Currency").agg(
    F.mean("Amount Received").alias("Promedio"),
    F.stddev("Amount Received").alias("Desviación estándar"),
    F.min("Amount Received").alias("Mínimo"),
    F.max("Amount Received").alias("Máximo")
)

# Calcular el total general (sin agrupamiento)
monsg = df.agg(
    F.lit("Total").alias("Payment Currency"),  # Identificador para el total
    F.mean("Amount Received").alias("Promedio"),
    F.stddev("Amount Received").alias("Desviación estándar"),
    F.min("Amount Received").alias("Mínimo"),
    F.max("Amount Received").alias("Máximo")
)

# Combinar las estadísticas por grupo con el total general
monsr= mons.union(monsg)

# Mostrar el resultado
#monsr.coalesce(1).write.csv('output/descripcionmonedas')

pcc = df.groupBy('Payment Currency').count()

pfc = df.groupBy('Payment Format').count()

#pcc.coalesce(1).write.csv('output/piemonedas')

#pfc.coalesce(1).write.csv('output/piepagos')


fecc = df.groupBy('TimeStamp').count()

#fecc.coalesce(1).write.csv('output/linefechas')


#modelos para compararlos
#hacer un random forest

dfrf = df.select('Amount Received', 'Payment Currency', 'Payment Format','SameBank','SameAccount','Is Laundering')

indexer1 = StringIndexer(inputCol='Payment Currency', outputCol='Currency')
indexer2 = StringIndexer(inputCol='Payment Format', outputCol='Format')

dfrf = indexer1.fit(dfrf).transform(dfrf)
dfrf = indexer2.fit(dfrf).transform(dfrf)

dfrf.show()

dfrf = dfrf.select('Amount Received', 'Currency', 'Format','SameBank','SameAccount','Is Laundering')

ass = VectorAssembler(inputCols = ['Amount Received', 'Currency', 'Format','SameBank','SameAccount'], outputCol = 'features')

dfrf = ass.transform(dfrf)

dfrf.show()

train_data, test_data = df.randomSplit([0.7, 0.3], seed=1998)

rf = RandomForestClassifier(labelCol="Is Laundering", featuresCol="features", numTrees=100)

rf_model = rf.fit(train_data)

rf_predictions = rf_model.transform(test_data)




#hacer un gboost

gbt = GBTClassifier(labelCol="Is Laundering", featuresCol="features", numTrees=100)

gbt_model = gbt.fit(train_data)

gbt_predictions = gbt_model.transform(test_data)


#comparar los dos


