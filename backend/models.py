from sqlalchemy import Column, Integer, String, Float, DateTime
import datetime
from database import Base

class ViolationRecord(Base):
    __tablename__ = "violations"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    violation_type = Column(String, index=True)
    severity = Column(String, index=True)
    rider_count = Column(Integer)
    plate_number = Column(String, index=True)
    confidence = Column(Float)
    image_url = Column(String)
