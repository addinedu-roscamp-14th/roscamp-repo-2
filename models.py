from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class Admin(db.Model):
    __tablename__ = 'admins'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Service(db.Model):
    __tablename__ = 'services'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    duration = db.Column(db.Integer, nullable=False)
    color = db.Column(db.String(7), nullable=False, default='#3498db')

class Workspace(db.Model):
    __tablename__ = 'workspaces'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)

reservation_services = db.Table('reservation_services',
    db.Column('reservation_id', db.Integer, db.ForeignKey('reservations.id', ondelete='CASCADE'), primary_key=True),
    db.Column('service_id', db.Integer, db.ForeignKey('services.id', ondelete='CASCADE'), primary_key=True)
)

class Reservation(db.Model):
    __tablename__ = 'reservations'
    id = db.Column(db.Integer, primary_key=True)
    customer_name = db.Column(db.String(64), nullable=False)
    customer_phone = db.Column(db.String(20), nullable=False)
    car_number = db.Column(db.String(20), nullable=False)
    memo = db.Column(db.Text, nullable=True)
    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime, nullable=False)
    workspace_id = db.Column(db.Integer, db.ForeignKey('workspaces.id'), nullable=False)

    workspace = db.relationship('Workspace', backref=db.backref('reservations', lazy=True))
    services = db.relationship('Service', secondary=reservation_services, backref=db.backref('reservations', lazy=True))

class SystemSetting(db.Model):
    __tablename__ = 'system_settings'
    id = db.Column(db.Integer, primary_key=True)
    setting_key = db.Column(db.String(64), unique=True, nullable=False)
    setting_value = db.Column(db.String(256), nullable=False)

class DeletedReservation(db.Model):
    __tablename__ = 'deleted_reservations'
    id = db.Column(db.Integer, primary_key=True)
    res_id = db.Column(db.String(50), nullable=False)
    customer_name = db.Column(db.String(64), nullable=False)
    customer_phone = db.Column(db.String(20), nullable=False)
    car_number = db.Column(db.String(20), nullable=False)
    start_time = db.Column(db.DateTime, nullable=False)
    delete_reason = db.Column(db.Text, nullable=False)
    admin_username = db.Column(db.String(64), nullable=False)
    deleted_at = db.Column(db.DateTime, default=datetime.utcnow)