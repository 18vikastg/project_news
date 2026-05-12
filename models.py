from flask_login import UserMixin
from database import get_db_connection

class User(UserMixin):
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email

    @staticmethod
    def get(user_id):
        conn = get_db_connection()
        user = conn.execute(
            'SELECT * FROM users WHERE id = ?', (user_id,)
        ).fetchone()
        conn.close()
        
        if not user:
            return None
        return User(id=user['id'], username=user['username'], email=user['email'])

    @staticmethod
    def find_by_username(username):
        conn = get_db_connection()
        user = conn.execute(
            'SELECT * FROM users WHERE username = ?', (username,)
        ).fetchone()
        conn.close()
        
        if not user:
            return None
        return User(id=user['id'], username=user['username'], email=user['email'])