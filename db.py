import sqlalchemy
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer,String,DateTime
from sqlalchemy.ext import declarative
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Image(Base):
    __tablename__ ='images'

    id = Column(Integer, primary_key=True)
    filename = Column(String)
    extension = Column(String)
    filepath = Column(String)
    uploader = Column(String,default='admin')
    created_on = Column(DateTime, default=datetime.now)

    def __str__(self):
        return self.filename

class Video(Base):
    __tablename__ ='videos'

    id = Column(Integer, primary_key=True)
    filename = Column(String)
    extension = Column(String)
    filepath = Column(String)
    uploader = Column(String,default='admin')
    created_on = Column(DateTime, default=datetime.now)

    def __str__(self):
        return self.filename

if __name__ == "__main__":
    engine = create_engine('sqlite:///db.sqlite3')
    Base.metadata.create_all(engine)