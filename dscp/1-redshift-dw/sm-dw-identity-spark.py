import pyspark.sql.functions as F

# Part1. This is for changing multiple device info to abbreviation type.

df = df.withColumn(
    'deviceinfo',
    F.when(
        F.col('deviceinfo').like('%ios%'),
        F.lit('ios')
    ).otherwise(F.when(
        F.col('deviceinfo').like('%huawei%'),
        F.lit('huawei')
    ).otherwise(F.when(
        F.col('deviceinfo').like('%htc%'),
        F.lit('htc')
    ).otherwise(F.when(
        F.col('deviceinfo').like('%lg%'),
        F.lit('lg')
    ).otherwise(F.when(
        F.col('deviceinfo').like('%lenovo%'),
        F.lit('lenovo')
    ).otherwise(F.when(
        F.col('deviceinfo').like('%macos%'),
        F.lit('macos')
    ).otherwise(F.when(
        F.col('deviceinfo').like('%moto%'),
        F.lit('moto')
    ).otherwise(F.when(
        F.col('deviceinfo').like('%samsung%'),
        F.lit('samsung')
    ).otherwise(F.when(
        F.col('deviceinfo').like('%sm%'),
        F.lit('sm')
    ).otherwise(F.when(
        F.col('deviceinfo').like('%trident%'),
        F.lit('trident')
    ).otherwise(F.when(
        F.col('deviceinfo').like('%windows%'),
        F.lit('windows')
    ).otherwise(F.when(
        F.col('deviceinfo').like('%rv%'),
        F.lit('rv')
    ))))))))))))
)

# Part2. This is for changing true/false to 1/0

df = df.withColumn('id_05_indicator', regexp_replace('id_05_indicator', 'true', '1'))
df = df.withColumn('id_05_indicator', regexp_replace('id_05_indicator', 'false', '0'))

df = df.withColumn('id_06_indicator', regexp_replace('id_06_indicator', 'true', '1'))
df = df.withColumn('id_06_indicator', regexp_replace('id_06_indicator', 'false', '0'))


df = df.withColumn('id_07_indicator', regexp_replace('id_07_indicator', 'true', '1'))
df = df.withColumn('id_07_indicator', regexp_replace('id_07_indicator', 'false', '0'))


df = df.withColumn('id_08_indicator', regexp_replace('id_08_indicator', 'true', '1'))
df = df.withColumn('id_08_indicator', regexp_replace('id_08_indicator', 'false', '0'))

df = df.withColumn('id_09_indicator', regexp_replace('id_09_indicator', 'true', '1'))
df = df.withColumn('id_09_indicator', regexp_replace('id_09_indicator', 'false', '0'))

df = df.withColumn('id_10_indicator', regexp_replace('id_10_indicator', 'true', '1'))
df = df.withColumn('id_10_indicator', regexp_replace('id_10_indicator', 'false', '0'))


df = df.withColumn('id_13_indicator', regexp_replace('id_13_indicator', 'true', '1'))
df = df.withColumn('id_13_indicator', regexp_replace('id_13_indicator', 'false', '0'))


df = df.withColumn('id_14_indicator', regexp_replace('id_14_indicator', 'true', '1'))
df = df.withColumn('id_14_indicator', regexp_replace('id_14_indicator', 'false', '0'))

df = df.withColumn('id_17_indicator', regexp_replace('id_17_indicator', 'true', '1'))
df = df.withColumn('id_17_indicator', regexp_replace('id_17_indicator', 'false', '0'))

df = df.withColumn('id_18_indicator', regexp_replace('id_18_indicator', 'true', '1'))
df = df.withColumn('id_18_indicator', regexp_replace('id_18_indicator', 'false', '0'))


df = df.withColumn('id_19_indicator', regexp_replace('id_19_indicator', 'true', '1'))
df = df.withColumn('id_19_indicator', regexp_replace('id_19_indicator', 'false', '0'))

df = df.withColumn('id_20_indicator', regexp_replace('id_20_indicator', 'true', '1'))
df = df.withColumn('id_20_indicator', regexp_replace('id_20_indicator', 'false', '0'))

df = df.withColumn('proxy_indicator', regexp_replace('proxy_indicator', 'true', '1'))
df = df.withColumn('proxy_indicator', regexp_replace('proxy_indicator', 'false', '0'))
