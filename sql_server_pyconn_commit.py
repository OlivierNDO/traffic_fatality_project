# Import Packages
#########################################################################################################################
import pandas as pd, numpy as np
import pyodbc
from sqlalchemy import create_engine, MetaData, Table, select

# Database Connection
#########################################################################################################################
def my_create_engine(mydsn = '<dsn>', mydatabase = '<db_name>', **kwargs):
    conn = pyodbc.connect(
    r'DSN=<dsn>;'
    r'UID=<username>;'
    r'PWD=<password>;'
    )
    cursor = conn.cursor()
    connection_string = 'mssql+pyodbc://@%s' % mydsn
    cargs = {'database': mydatabase}
    cargs.update(**kwargs)
    e = create_engine(connection_string, connect_args=cargs)
    return e, cursor, conn

def tbl_write(tbl_name, engine, pandas_df):
    pandas_df.to_sql(tbl_name, engine, if_exists = 'append', chunksize = None, index = False)

eng, cursor, conn = my_create_engine()

"""
- Table Write Example:

temp_dat = pd.DataFrame({'y_field': range(100), 'x_field': range(100)})
tbl_write('temp_tbl', eng, temp_dat)


- Query Execution Example
df = pd.read_sql("select y_field, x_field from dbo.temp_tbl", eng.connect())


"""
