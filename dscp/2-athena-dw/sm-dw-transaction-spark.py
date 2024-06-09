import pyspark.sql.functions as F

# Part1. This is for quickly changing m1/2/3/5/6/7/8/9 to missing indicator and one-hot encoding

col_names_lst = ['m1','m2','m3','m5','m6','m7','m8','m9']
for x in col_names_lst: 
    df = df.withColumn(
        x + '_indicator',
        F.when((F.col(x) == 'T') | (F.col(x) == 'F') , 0).otherwise(1)
    )

    df = df.withColumn(
        x + '_T',
        F.when((F.col(x) == 'T') , 1).otherwise(0)
    )

    df = df.withColumn(
        x + '_F',
        F.when((F.col(x) == 'F') , 1).otherwise(0)
    )

    df = df.drop(x)

# Part2. Using custom analysis to check missing values counts in df
# Table is available as variable `df` of pandas dataframe
# Output Altair chart is available as variable `chart`

# this is for all columns in df
import pandas as pd
import altair as alt

df_alt = pd.DataFrame(df.columns, columns = ['column_name'])
df_alt['missing_cnt'] = 0

for i in range(len(df.index)):
    df_alt.loc[i:i,'missing_cnt'] = df.iloc[i].isnull().sum()

df_alt = df_alt[["column_name","missing_cnt"]]

base = alt.Chart(df_alt)
bar = base.mark_bar().encode(x='column_name',y='missing_cnt')

chart = bar

# this is for designated columns in df
import pandas as pd
import altair as alt

designated_columns = ['transactionid','isfraud','transactiondt','transactionamt','card1','card2','card3','card5']

df_alt = pd.DataFrame(designated_columns, columns = ['column_name'])
df_alt['missing_cnt'] = 0

for i in range(len(designated_columns)):
    df_alt.loc[i:i,'missing_cnt'] = df[designated_columns[i]].isnull().sum()

df_alt = df_alt[["column_name","missing_cnt"]]

base = alt.Chart(df_alt)
bar = base.mark_bar().encode(x='column_name',y='missing_cnt')

chart = bar


# Part3. This is for changing d2-d15 with indicator and medium value for missing value

from pyspark.ml.feature import Imputer
import pyspark.sql.functions as F

col_names_lst = ['d'+str(i) for i in range(2,16)]

for x in col_names_lst:
    df = df.withColumn(
        x + '_indicator',
        F.when((F.col(x).isNull()) , 1).otherwise(0)
    )

    imputer = Imputer(
        inputCols = [x],
        outputCols = [x + "_imputed"]
        ).setStrategy("median")

    # Add imputation cols to df
    df = imputer.fit(df).transform(df)
    df = df.drop(x)

# Part4. This is for changing v1-v339 with indicator and zero for missing value

from pyspark.ml.feature import Imputer
import pyspark.sql.functions as F

col_names_lst = ['v'+str(i) for i in range(1,340)]

for x in col_names_lst:
    df = df.withColumn(
        x + '_indicator',
        F.when((F.col(x).isNull()) , 1).otherwise(0)
    )

    imputer = Imputer(
        inputCols = [x],
        outputCols = [x + "_imputed"]
        ).setStrategy("median")

    # Add imputation cols to df
    df = imputer.fit(df).transform(df)
    df = df.drop(x)

# Part5. This is for changing true/false to 1/0
from pyspark.sql.functions import regexp_replace

df = df.withColumn('p_emaildomain_indicator', regexp_replace('p_emaildomain_indicator', 'true', '1'))
df = df.withColumn('p_emaildomain_indicator', regexp_replace('p_emaildomain_indicator', 'false', '0'))

df = df.withColumn('r_emaildomain_indicator', regexp_replace('r_emaildomain_indicator', 'true', '1'))
df = df.withColumn('r_emaildomain_indicator', regexp_replace('r_emaildomain_indicator', 'false', '0'))

df = df.withColumn('m4_indicator', regexp_replace('m4_indicator', 'true', '1'))
df = df.withColumn('m4_indicator', regexp_replace('m4_indicator', 'false', '0'))

df = df.withColumn('card2_indicator', regexp_replace('card2_indicator', 'true', '1'))
df = df.withColumn('card2_indicator', regexp_replace('card2_indicator', 'false', '0'))

df = df.withColumn('card5_indicator', regexp_replace('card5_indicator', 'true', '1'))
df = df.withColumn('card5_indicator', regexp_replace('card5_indicator', 'false', '0'))

df = df.withColumn('addr1_indicator', regexp_replace('addr1_indicator', 'true', '1'))
df = df.withColumn('addr1_indicator', regexp_replace('addr1_indicator', 'false', '0'))

df = df.withColumn('addr2_indicator', regexp_replace('addr2_indicator', 'true', '1'))
df = df.withColumn('addr2_indicator', regexp_replace('addr2_indicator', 'false', '0'))

df = df.withColumn('dist1_indicator', regexp_replace('dist1_indicator', 'true', '1'))
df = df.withColumn('dist1_indicator', regexp_replace('dist1_indicator', 'false', '0'))