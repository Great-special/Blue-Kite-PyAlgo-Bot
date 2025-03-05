import sqlite3



# Create the table to store used order blocks
def create_table():
    # Connect to the SQLite3 database (or create it if it doesn't exist)
    conn = sqlite3.connect('order_blocks.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS used_order_blocks (
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            order_block_type TEXT,
            high REAL,
            low REAL,
            timestamp TEXT,
            status TEXT
        )
    ''')

    conn.commit()
    
    conn.close()
    



def insert_order_block(symbol, order_block_type, high, low, timestamp, status):
    conn = sqlite3.connect('order_blocks.db')
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO used_order_blocks (symbol, order_block_type, high, low, timestamp, status)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (symbol, order_block_type, high, low, timestamp, status))
    conn.commit()
    conn.close()
    


def get_used_order_blocks(symbol):
    conn = sqlite3.connect('order_blocks.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM used_order_blocks WHERE symbol = ?', (symbol,))
    data = cursor.fetchall()
    conn.close()
    return data
