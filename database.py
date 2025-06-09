import mysql.connector

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",  # Change this to your MySQL password
        database="customer_satisfaction"
    )
