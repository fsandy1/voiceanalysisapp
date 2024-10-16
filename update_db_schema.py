import sqlite3

DATABASE = "database.db"

def update_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    # Add 'identifier' column to 'VoiceResponses' table if it doesn't already exist
    try:
        cursor.execute("ALTER TABLE VoiceResponses ADD COLUMN identifier TEXT")
        print("Database updated successfully.")
    except sqlite3.OperationalError as e:
        print(f"OperationalError: {e} - It may already exist.")
    conn.commit()
    conn.close()

if __name__ == "__main__":
    update_db()
