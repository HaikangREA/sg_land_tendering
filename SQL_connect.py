# REFACTOR_COMPLETE

import psycopg2 as psycopg2
import pandas as pd
import logging
import json
import sqlalchemy
import sys
from io import StringIO
import random
from psycopg2.errors import DuplicateTable, DuplicateColumn

from rea_python.constants import OutputFormat, DBCopyMode


log = logging.getLogger(__name__)

# %%

NEED_TO_CREATE_DB_CONFIG_FILE = False
if NEED_TO_CREATE_DB_CONFIG_FILE:
    myconfig = {
        "host": "alchemist.cermeo8wsjdb.ap-southeast-1.rds.amazonaws.com",
        "database": "postgres",
        "port": 5432,
        "user": "data_team",
        "password": "password_here",
    }
    with open("dbconfig.txt", "w") as f:
        json.dump(myconfig, f)

# %%


class DBConnection:
    def __init__(self, **kwargs):
        if not len(kwargs):

            default_dbconfig_path = "dbconfig.txt"  # at command execution directory

            log.warning(
                f"No db conn args given, searching for DB "
                f"credentials from file at path: {default_dbconfig_path}"
            )

            try:
                with open(default_dbconfig_path, "r") as f:
                    kwargs = json.load(f)

            except FileNotFoundError:
                log.warning(
                    "Unable to locate dbconfig.txt, most likely in production. "
                    "Trying to access AWS Secret Manager. "
                )

                # add import statement here because rea-python-common is only needed here
                # The following code is entirely used for case where dbconfig.txt is not needed
                # It aims to provide three things: the SQLalchemy engine, psycopg2 conn and cursor
                # It was built so the other part of the code logic will not be impacted.
                from rea_python.main.aws import get_secret
                from rea_python.main.database import PostgresHook

                engine_config = get_secret(
                    "legacy/data_team/db_conn_uri"
                )  # this is actually the engine uri
                hook = PostgresHook()
                hook.set_conn_from_uri(engine_config)

                # hard coding so to keep the key name unchanged
                kwargs.update(
                    {
                        "host": "alchemist.cermeo8wsjdb.ap-southeast-1.rds.amazonaws.com",
                        "database": "postgres",
                        "port": 5432,
                        "user": "data_team",
                        "password": hook.conn_info.get("password"),
                    }
                )
            else:
                log.error(
                    "Unable to find either dbconfig.txt or AWS credentials, please check. "
                )
                sys.exit(1)

        # The alchemy engine, used by some libaries
        engine_config = "postgresql+psycopg2://%s:%s@%s/%s" % (
            kwargs["user"],
            kwargs["password"],
            kwargs["host"],
            kwargs["database"],
        )

        self.conn = psycopg2.connect(**kwargs)
        self.cur = self.conn.cursor()
        self.engine = sqlalchemy.create_engine(engine_config)
        self.engine_conn = self.engine.connect()

    # Reads about 1050 rows per second on benchmark
    def read_data(self, sql_command, logcommand=True):
        """Converts an sql select into a pandas df"""
        if logcommand:
            log.debug(sql_command)
            print(sql_command)
        return pd.read_sql(sql_command, self.conn)

    def read_data_selection(self, tablename, df, col, query="select *", coltype="uuid"):
        """Loads part of a table, selecting keys from a list given as a dataframe
        tablename = table from which to select
        df -- dataframe with key values (series if selecction on only one columns)
        col -- key or list of keys for for filtering
        coltype -- tpye or list of types corresponding to the above keys
        query -- sql query used for selection
        """
        # Tests indicated that WHERE is much faster than JOIN for a selection
        # make sure the col, coltype are lists
        if type(col) != list:
            col = [col]
        if type(coltype) != list:
            coltype = [coltype]
        assert len(col) == len(
            coltype
        ), "read_data_selection: col and coltype must have same length"

        # Set a temporary table sufix
        temp_sufix = "_temp%d" % random.randint(1000, 10000)

        # Save the selection to a table
        sql_columns = ",\n".join(
            ["{} {} NOT NULL".format(n, t) for n, t in zip(col, coltype)]
        )
        self.sql_code(
            """drop table if exists {0}{2};
            create table {0}{2} ({1});""".format(
                tablename, sql_columns, temp_sufix
            )
        )
        self.copy_from_stringio(df, "{}{}".format(tablename, temp_sufix))
        # Load filtered batch from distances
        sql_columns = "(" + ",".join(col) + ")"
        res = self.read_data(
            """{0} from {1} where {2} in
            (select * from {1}{3})""".format(
                query, tablename, sql_columns, temp_sufix
            )
        )
        self.drop_table("{}{}".format(tablename, temp_sufix))
        return res

    # Use Read_data instead, much faster
    def load_data(self, sql_command, logcommand=True):
        """Uses fetchall of an sql command"""
        print("Deprecated: use sql_code")
        return self.sql_code(sql_command)

    def sql_code(self, sql_command, logcommand=True):
        """Executes command and return the result as a list of tuples
        If the result is a single number (eg table count) use result[0][0] to
        get thee value
        """
        if self.isexception():
            self.rollback()
        # Code contains many individual row updates - by default are excluded from log
        if logcommand:
            log.debug(sql_command)
            print(sql_command)
        self.cur.execute(sql_command)
        result = None
        if self.cur.description:
            result = self.cur.fetchall()
        self.conn.commit()
        return result

    def rollback(self):
        # Rollback a failed query in case of error:
        self.conn.rollback()

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.commit()
        self.cur.close()
        self.conn.close()

    def isexception(self):
        return self.conn.status == psycopg2.extensions.STATUS_IN_TRANSACTION

    def drop_table(self, table_name):
        sql_command = "DROP TABLE IF EXISTS " + table_name + " CASCADE;"
        self.sql_code(sql_command)

    def insert_sql(self, table_name, insert_data, logcommand=True):
        # Replaces all values NaN valeus with 'Null' before insertion
        insert_data = [e if not pd.isnull(e) else None for e in insert_data]
        sql_command = (
            "INSERT INTO "
            + table_name
            + " VALUES ("
            + "%s," * (len(insert_data) - 1)
            + "%s)"
        )
        if logcommand:
            print(sql_command)
            log.debug(sql_command)
        self.cur.execute(sql_command, insert_data)
        self.conn.commit()

    def copy_from_stringio(self, df, table, logcommand=True, sep="|"):
        """Quickly insert all dataframe rows into a table
        If the table exists appends the dataframe to the table
        Saves the dataframe in memory  and uses copy_from() to copy it to the table
        """
        # copy_from has a problem with the separator, make sure the CSV separater is
        # not inside the string values inserted into the table
        try:
            # save dataframe to an in memory buffer
            buffer = StringIO()
            # take care of the Null values:
            ndf = df.where(df.notnull(), "\\N")
            # ndf = df
            ndf.to_csv(buffer, sep=sep, index=False, header=False)
            buffer.seek(0)

            message = "Copy_from_stringio into table %s" % table
            if logcommand:
                print(message)
                log.debug(message)
            self.cur.copy_from(buffer, table, sep=sep, null="\\N")
            self.conn.commit()
        except Exception as e:
            # df.to_csv('temp.csv', index = False)
            raise (e)

    # Slower. Use copy_from_stringio instead
    def df_to_table(self, df, table_name):
        """Saves dataframe to sql table
        Drops table if exists
        """
        message = "Save Dataframe as table: %s" % table_name
        print(message)
        log.debug(message)
        # drop table if exists
        self.drop_table(table_name)
        # pandas.to_sql does not automatically interpret table name
        # check for schema in the name:
        schema_split = table_name.split(".")
        if len(schema_split) > 1:
            schema = schema_split[0]
            table = ".".join(schema_split[1:])
            df.to_sql(table, self.engine, schema=schema)
        else:
            df.to_sql(table_name, self.engine)
        self.conn.commit()

    def create_table(self, table_name, table_inf, drop=True):
        """table_inf must enclose table information in ()
        if table_inf is a dataframe the DDL information is extracted automatically
        """
        if type(table_inf) == pd.core.frame.DataFrame:
            sql_info = pd.io.sql.get_schema(table_inf, "dummy")
            table_inf = sql_info[sql_info.find("(") :]
        if drop:
            self.drop_table(table_name)
        sql_command = "CREATE TABLE " + table_name + table_inf
        print(sql_command)
        log.debug(sql_command)
        try:
            self.cur.execute(sql_command)
            self.conn.commit()
        except DuplicateTable as e:
            print("Table creation skipped:", e)
            self.rollback()
            return -1

    def rename_table(self, src, dest):
        """Rename table src to dest"""
        src_sc, src_name = src.split(".")
        dest_sc, dest_name = dest.split(".")
        # Drop dest table
        self.drop_table(dest)
        if src_sc != dest_sc:
            # transfer table from one schema to another
            self.sql_code(
                """ALTER SCHEMA {0}
                TRANSFER {1}.{2}""".format(
                    dest_sc, src_sc, src_name
                )
            )
        self.sql_code(
            """ALTER TABLE {}.{}
                RENAME TO {};""".format(
                dest_sc, src_name, dest_name
            )
        )

    def update_table(self, dest_table, src_table, idcol, cols):
        """Updates all columns 'cols' in table 'dest_table' from 'src_table'
        based on the key 'idcol'
        """
        sql_columns = ",\n".join(["{0} = src.{0}".format(e) for e in cols])
        sql_command = """UPDATE {0} dest SET
            {1}
            FROM {2} src
            WHERE
            src.{3} = dest.{3};""".format(
            dest_table, sql_columns, src_table, idcol
        )
        self.sql_code(sql_command)

    def alter_col_type(self, table, col, coltype, convert_type=None):
        """Alter column type. Might fail if an automatic convertion is not possible
        Use covert_type if a direct conversion is not possible
        (eg bool::integer works for a bool > int2 type change)
        """
        if convert_type is None:
            convert_type = coltype
        sql_command = """alter table {0}
        alter column {1} type {2}
        using {1}::{3};""".format(
            table, col, coltype, convert_type
        )
        self.sql_code(sql_command)

    def copy_table(self, dest_table, src_table, copy_data=True):
        """Create a new table 'dest_table' like 'src_table', including similar constraints and
        indexes
        Copies all data into 'dest_table' if copy_data
        """
        sql_command = """drop table if exists {0} cascade;
        create table {0} (like {1} including indexes);
        """.format(
            dest_table, src_table
        )
        self.sql_code(sql_command)
        if copy_data:
            sql_command = """insert into {0} select * from {1};
            """.format(
                dest_table, src_table
            )
            self.sql_code(sql_command)

    def add_column(self, table, colname, coltype):
        """Add column, ignores error if column exists"""
        sql_command = """alter table {0} add {1} {2};""".format(table, colname, coltype)
        try:
            self.sql_code(sql_command)
        except DuplicateColumn as e:
            message = "Column already exists, ignore: " + str(e)
            log.debug(message)
            print(message)
            self.rollback()
            return -1

    def get_col_order(self, table):
        """Return table columns, in order. Use it to correctly upload data"""
        return self.read_data("select * from " + table + " limit 0;").columns

    def get_int_cols(self, table):
        """Get table integer columns. Nullable integer columns must be handled
        specially when uploading from pandas
        Expects a table name with schema
        """
        sc, tn = table.split(".")
        res = self.sql_code(
            """select column_name
            -- , data_type, character_maximum_length, column_default, is_nullable
            from INFORMATION_SCHEMA.COLUMNS where table_name = '{}'
            and table_schema = '{}'
            and data_type in ('bit','tinyint','smallint','integer',
                'bigint','decimal','numeric');""".format(
                tn, sc
            )
        )
        return [e[0] for e in res]

    # END of class DBConnection


class DBConnectionRS:
    def __init__(self):
        from rea_python.main.aws import get_secret
        from rea_python.main.database import RedshiftHook
        from rea_python.main.aws import S3Hook

        engine_config = get_secret(
            "prod/redshift/pipeline/db_conn_uri"
        )  # this is actually the engine uri
        IAM_ROLE_ARN = "arn:aws:iam::051694948699:role/prod-redshift-aws-access"
        REDSHIFT_HOOK_BRIDGE_FOLDER = "redshift-copy"
        REDSHIFT_HOOK_BRIDGE_BUCKET = "dev-misc-usage"
        IMAGE_S3_BUCKET = "rea-ml-model"
        self.hook = RedshiftHook(
            iam_role_arn=IAM_ROLE_ARN,
            via_s3_bucket=REDSHIFT_HOOK_BRIDGE_BUCKET,
            via_s3_folder=REDSHIFT_HOOK_BRIDGE_FOLDER,
        )
        self.hook.set_conn_from_uri(engine_config)
        self.s3 = S3Hook(IMAGE_S3_BUCKET)

    def read_data(self, sql_command, logcommand=True):
        """Converts an sql select into a pandas df"""
        if logcommand:
            log.debug(sql_command)
            print(sql_command)
        return self.hook.execute_raw_query(
            query=sql_command, output_format=OutputFormat.pandas
        )

    def sql_code(self, sql_command, logcommand=True, output=OutputFormat.raw):
        """Executes command and return the result as a list of tuples
        If the result is a single number (eg table count) use result[0][0] to
        get thee value
        """
        if logcommand:
            log.debug(sql_command)
            print(sql_command)
        return self.hook.execute_raw_query(query=sql_command, output_format=output)

    def copy_from_stringio(self, df, table, logcommand=True, ddl=None):
        """
        Quickly insert all dataframe rows into a table
        If the table exists appends the dataframe to the table
        """
        try:
            message = "Copy_from_stringio into table %s" % table
            if logcommand:
                print(message)
                log.debug(message)
            self.hook.copy_from_df(
                df=df, target_table=table, mode=DBCopyMode.APPEND, ddl=ddl
            )

        except Exception as e:
            # df.to_csv('temp.csv', index = False)
            raise (e)

    def copy_from_df(self, df, table, logcommand=True, ddl=None):
        """
        Quickly insert all dataframe rows into a DB table
        """
        try:
            message = "Copy_from_df into table %s" % table
            if logcommand:
                print(message)
                log.debug(message)
            self.hook.copy_from_df(
                df=df,
                target_table=table,
                mode=DBCopyMode.DROP,
                ddl=ddl,
                con=self.hook.sqla_engine,
            )

        except Exception as e:
            # df.to_csv('temp.csv', index = False)
            raise (e)

    def drop_table(self, table_name):
        sql_command = "DROP TABLE IF EXISTS " + table_name + " CASCADE;"
        self.sql_code(sql_command)

    def insert_sql(self, table_name, insert_data, logcommand=True):
        # Replaces all values NaN values with 'Null' before insertion
        insert_data = [e if not pd.isnull(e) else None for e in insert_data]
        sql_command = (
            "INSERT INTO "
            + table_name
            + " VALUES ("
            + "%s," * (len(insert_data) - 1)
            + "%s)"
        )
        if logcommand:
            print(sql_command)
            log.debug(sql_command)
        self.sql_code(sql_command)

    def read_data_selection(self, tablename, df, col, query="select *", coltype="uuid"):
        """Loads part of a table, selecting keys from a list given as a dataframe
        tablename = table from which to select
        df -- dataframe with key values (series if selecction on only one columns)
        col -- key or list of keys for for filtering
        coltype -- tpye or list of types corresponding to the above keys
        query -- sql query used for selection
        """
        # Tests indicated that WHERE is much faster than JOIN for a selection
        # make sure the col, coltype are lists
        if type(col) != list:
            col = [col]
        if type(coltype) != list:
            coltype = [coltype]
        assert len(col) == len(
            coltype
        ), "read_data_selection: col and coltype must have same length"

        # Set a temporary table sufix
        temp_sufix = "_temp%d" % random.randint(1000, 10000)
        # Save the selection to a table
        sql_columns = ",\n".join(
            ["{} {} NOT NULL".format(n, t) for n, t in zip(col, coltype)]
        )
        self.sql_code(
            """drop table if exists {0}{2};
                    create table {0}{2} ({1});""".format(
                tablename, sql_columns, temp_sufix
            )
        )
        self.copy_from_stringio(df, "{}{}".format(tablename, temp_sufix))
        # Load filtered batch from distances
        sql_columns = "(" + ",".join(col) + ")"
        res = self.read_data(
            """{0} from {1} where {2} in
                    (select {2} from {1}{3})""".format(
                query, tablename, sql_columns, temp_sufix
            )
        )
        self.drop_table("{}{}".format(tablename, temp_sufix))
        return res

    def create_table(self, table_name, table_inf, drop=True):
        """table_inf must enclose table information in ()
        if table_inf is a dataframe the DDL information is extracted automatically
        """
        if type(table_inf) == pd.core.frame.DataFrame:
            sql_info = pd.io.sql.get_schema(table_inf, "dummy")
            table_inf = sql_info[sql_info.find("(") :]
        if drop:
            self.drop_table(table_name)
        sql_command = "CREATE TABLE IF NOT EXISTS " + table_name + table_inf
        log.debug(sql_command)

        self.sql_code(sql_command)

    #        except DuplicateTable as e:
    #            print('Table creation skipped:', e)
    #            self.rollback()
    #            return -1

    def rename_table(self, src, dest):
        """Rename table src to dest"""
        src_sc, src_name = src.split(".")
        dest_sc, dest_name = dest.split(".")
        # Drop dest table
        self.drop_table(dest)
        if src_sc != dest_sc:
            # transfer table from one schema to another
            self.sql_code(
                """ALTER SCHEMA {0}
                    TRANSFER {1}.{2}""".format(
                    dest_sc, src_sc, src_name
                )
            )
        self.sql_code(
            """ALTER TABLE {}.{}
                    RENAME TO {};""".format(
                dest_sc, src_name, dest_name
            )
        )

    def update_table(self, dest_table, src_table, idcol, cols):
        """Updates all columns 'cols' in table 'dest_table' from 'src_table'
        based on the key 'idcol'
        """
        sql_columns = ",\n".join(["{0} = src.{0}".format(e) for e in cols])
        sql_command = """UPDATE {0} dest SET
                {1}
                FROM {2} src
                WHERE
                src.{3} = dest.{3};""".format(
            dest_table, sql_columns, src_table, idcol
        )
        self.sql_code(sql_command)

    def alter_col_type(
        self, table, col, coltype, convert_type=None
    ):  # NOTE: ALTER TABLE ALTER COLUMN cannot run inside a transaction block
        """Alter column type. Might fail if an automatic convertion is not possible
        Use covert_type if a direct conversion is not possible
        (eg bool::integer works for a bool > int2 type change)
        """
        if convert_type is None:
            convert_type = coltype
        sql_command = """ alter table {0} add column {1}_new {2};
                update {0} set {1}_new = cast({1} as {2});
                alter table {0} drop column {1};
                alter table {0} rename column {1}_new to {1};""".format(
            table, col, coltype, convert_type
        )
        self.sql_code(sql_command)

    def copy_table(self, dest_table, src_table, copy_data=True):
        ## removed include indexes
        """Create a new table 'dest_table' like 'src_table', including similar constraints and
        indexes
        Copies all data into 'dest_table' if copy_data
        """
        sql_command = """drop table if exists {0} cascade;
            create table {0} (like {1});
            """.format(
            dest_table, src_table
        )
        self.sql_code(sql_command)
        if copy_data:
            sql_command = """insert into {0} select * from {1};
                """.format(
                dest_table, src_table
            )
            self.sql_code(sql_command)

    def add_column(self, table, colname, coltype):
        """Add column, ignores error if column exists"""
        sql_command = """alter table {0} add {1} {2};""".format(table, colname, coltype)
        try:
            self.sql_code(sql_command)
        except sqlalchemy.exc.ProgrammingError as e:
            log.debug(e)
            print(e)

    #            message = 'Column already exists, ignore: ' + str(e)
    #            log.debug(message)
    #            print(message)
    #            self.rollback()
    #            return -1

    def get_col_order(self, table):
        """Return table columns, in order. Use it to correctly upload data"""
        return self.read_data("select * from " + table + " limit 1;").columns

    def get_int_cols(self, table):
        """Get table integer columns. Nullable integer columns must be handled
        specially when uploading from pandas
        Expects a table name with schema
        """
        sc, tn = table.split(".")
        res = self.sql_code(
            """select column_name
                -- , data_type, character_maximum_length, column_default, is_nullable
                from INFORMATION_SCHEMA.COLUMNS where table_name = '{}'
                and table_schema = '{}'
                and data_type in ('bit','tinyint','smallint','integer',
                    'bigint','decimal','numeric');""".format(
                tn, sc
            )
        )
        return [e[0] for e in res]
