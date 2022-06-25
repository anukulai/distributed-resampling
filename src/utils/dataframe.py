from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import row_number, monotonically_increasing_id
from pyspark.sql.types import NumericType


def init_df(schema):
    return SparkSession.getActiveSession().createDataFrame([], schema)


def index_df(df, index_col):
    return df.withColumn(index_col, row_number().over(Window.orderBy(monotonically_increasing_id())) - 1)


def select_df(df, start, end):
    index_col = 'index'
    df = index_df(df, index_col)

    return df.filter((df[index_col] >= start) & (df[index_col] <= end)).drop(index_col)


def collect_col(df, col):
    return df.select(col).rdd.map(lambda row: row[col]).collect()


def collect_cols(df, cols):
    return df.select(*cols).rdd.map(lambda row: [row[col] for col in cols]).collect()


def get_cat_cols(df):
    return [field.name for field in df.schema.fields if not isinstance(field.dataType, NumericType)]


def get_num_cols(df):
    return [field.name for field in df.schema.fields if isinstance(field.dataType, NumericType)]
