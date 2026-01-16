import sqlite3

def connect_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    # create user table if not exists
    
    user_table_query = """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username varchar(255) NOT NULL,
        email varchar(255) NOT NULL UNIQUE,
        password varchar(255) NOT NULL
        );
    """
    c.execute(user_table_query)
    
    save = conn.commit
    close = conn.close
    return c, save, close


def create_prediction_table():
    query = """
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        data_bucket_id INTEGER NOT NULL UNIQUE,
        prediction_id varchar(255) NOT NULL UNIQUE,
        title varchar(255) NOT NULL
        );
    """
    
    c, save, close = connect_db()
    # drop prediction table
    c.execute("DROP TABLE predictions")
    c.execute(query)
    
    save()
    close()

def create_job_table():
    query = """
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        job_id varchar(255) NOT NULL UNIQUE,
        bucket_id varchar(255) NOT NULL,
        status varchar(255),
        start_date varchar(255),
        end_date varchar(255),
        criteria varchar(255)
        );
    """
    c, save, close = connect_db()
    # drop prediction table
    c.execute("DROP TABLE jobs")
    c.execute(query)
    
    save()
    close()



# create_prediction_table()
# create_job_table()
