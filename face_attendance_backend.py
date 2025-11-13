from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime, timedelta
from passlib.context import CryptContext
import jwt
from dotenv import load_dotenv
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo



# Load environment variables from .env file
load_dotenv()
import os
import secrets
import string
import json  # used in register/checkin
import random
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, ForeignKey, Boolean, Text, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from google.cloud.sql.connector import Connector
import pymysql

ET = ZoneInfo("US/Eastern")
# ============= CONFIGURATION =============
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# Google Cloud SQL Configuration
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "your-password")
DB_NAME = os.getenv("DB_NAME", "face_attendance")
INSTANCE_CONNECTION_NAME = os.getenv("INSTANCE_CONNECTION_NAME", "project:region:instance")

# Face matching configuration
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.93"))  # stricter default
DUPLICATE_FACE_THRESHOLD = float(os.getenv("DUPLICATE_FACE_THRESHOLD", "0.90"))  # detect same face on signup

# ============= DATABASE SETUP =============
Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Initialize Cloud SQL Connector
connector = Connector()

def getconn() -> pymysql.connections.Connection:
    """Create database connection to Cloud SQL"""
    conn = connector.connect(
        INSTANCE_CONNECTION_NAME,
        "pymysql",
        user=DB_USER,
        password=DB_PASS,
        db=DB_NAME
    )
    return conn

# Create SQLAlchemy engine using Cloud SQL connector
engine = create_engine(
    "mysql+pymysql://",
    creator=getconn,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ============= MODELS =============
# Association table for student enrollments
enrollments = Table(
    'enrollments',
    Base.metadata,
    Column('student_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('course_id', Integer, ForeignKey('courses.id'), primary_key=True),
    Column('enrolled_at', DateTime, default=datetime.now(ET))
)

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False)  # 'student' or 'professor'
    student_id = Column(String(8), unique=True, index=True, nullable=True)  # 8-digit ID for students
    student_id_edited = Column(Boolean, default=False)  # Track if ID has been edited
    face_embedding = Column(Text, nullable=True)  # JSON string of face vector (L2-normalized)
    created_at = Column(DateTime, default=datetime.now(ET))

    # Relationships
    courses_taught = relationship("Course", back_populates="professor")
    enrolled_courses = relationship("Course", secondary=enrollments, back_populates="students")
    attendance_records = relationship("Attendance", back_populates="student")

class Course(Base):
    __tablename__ = "courses"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    code = Column(String(20), unique=True, index=True, nullable=False)
    professor_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.now(ET))
    
    # Relationships
    professor = relationship("User", back_populates="courses_taught")
    students = relationship("User", secondary=enrollments, back_populates="enrolled_courses")
    sessions = relationship("Session", back_populates="course")

class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey('courses.id'), nullable=False)
    start_time = Column(DateTime(timezone=True), default=lambda: datetime.now(ET))  # Add timezone=True
    end_time = Column(DateTime(timezone=True), nullable=True)  # Add timezone=True
    late_after_minutes = Column(Integer, default=5)
    absent_after_minutes = Column(Integer, default=15)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    course = relationship("Course", back_populates="sessions")
    attendance_records = relationship("Attendance", back_populates="session")

class Attendance(Base):
    __tablename__ = "attendance"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey('sessions.id'), nullable=False)
    student_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(ET))  # Add timezone=True
    status = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=True)
    
    # ... rest of the code
    
    # Relationships
    session = relationship("Session", back_populates="attendance_records")
    student = relationship("User", back_populates="attendance_records")

# Create all tables
Base.metadata.create_all(bind=engine)

# ============= PYDANTIC SCHEMAS =============
class UserRegister(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: str
    student_id: Optional[str] = None  # Now required for students
    face_embedding: Optional[List[float]] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class StudentIdUpdate(BaseModel):
    new_student_id: str

class CourseCreate(BaseModel):
    name: str

class CourseEnroll(BaseModel):
    join_code: str

class SessionStart(BaseModel):
    course_id: int
    late_after_minutes: int = 5
    absent_after_minutes: int = 15

class CheckIn(BaseModel):
    course_id: int
    face_embedding: List[float]

class CheckInWithStudentId(BaseModel):
    course_id: int
    student_id: str
    face_embedding: List[float]

class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    name: str
    student_id: Optional[str] = None  # Include student ID in token response

# ============= FASTAPI APP =============
# ============= FASTAPI APP =============

#CHANGES 
app = FastAPI(title="Face Attendance API")

# CORS middleware (tighten ALLOWED_ORIGINS in prod)
raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in raw_origins.split(",") if o.strip()]
# If wildcard, cannot set allow_credentials=True per browser rules
allow_creds = not (len(ALLOWED_ORIGINS) == 1 and ALLOWED_ORIGINS[0] == "*")

origins = [
    "https://attendousa.github.io",
    "https://attendo-ojjl.onrender.com"
]

# CORS Configuration - Must be added immediately after app creation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Preflight handler for OPTIONS requests
@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str):
    return {"message": "OK"}

security = HTTPBearer()

# ============= DEPENDENCY =============
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============= HELPER FUNCTIONS =============

def generate_join_code(length=6):
    """Generate random alphanumeric join code"""
    chars = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(chars) for _ in range(length))

def create_access_token(data: dict):
    """Create JWT token"""
    to_encode = data.copy()
    expire = datetime.now(ET) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def l2_normalize(vec: List[float]) -> List[float]:
    """L2-normalize a vector to unit length"""
    import math
    norm = math.sqrt(sum(v * v for v in vec))
    if not norm:
        return vec[:]  # avoid division by zero; upstream should ensure non-zero vectors
    return [v / norm for v in vec]

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors (expects normalized vectors)"""
    import math
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)

# ============= ROUTES =============
@app.get("/")
def root():
    return {"message": "Face Attendance API", "status": "running"}

@app.post("/auth/register", response_model=Token)
def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """Register new user with student-provided ID"""
    # Check if email exists
    existing = db.query(User).filter(User.email == user_data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Guard: students must provide a face
    if user_data.role == "student" and (not user_data.face_embedding or len(user_data.face_embedding) == 0):
        raise HTTPException(status_code=400, detail="Face photo/embedding required for student accounts")
    
    # Validate student ID for students
    student_id = None
    if user_data.role == "student":
        if not user_data.student_id:
            raise HTTPException(status_code=400, detail="Student ID is required for student accounts")
        
        # Validate format (8 digits)
        if not user_data.student_id.isdigit() or len(user_data.student_id) != 8:
            raise HTTPException(status_code=400, detail="Student ID must be exactly 8 digits")
        
        # Check if student ID is already taken
        existing_id = db.query(User).filter(User.student_id == user_data.student_id).first()
        if existing_id:
            raise HTTPException(status_code=400, detail="This Student ID is already registered")
        
        student_id = user_data.student_id

    # Hash password
    hashed = pwd_context.hash(user_data.password[:72])
    
    # Prepare face embedding (L2-normalized) and block duplicates
    face_str = None
    norm_emb = None
    if user_data.face_embedding:
        norm_emb = l2_normalize(user_data.face_embedding)

        # Duplicate-face check
        existing_with_face = db.query(User).filter(User.face_embedding.isnot(None)).all()
        for u in existing_with_face:
            try:
                other = json.loads(u.face_embedding)
            except Exception:
                continue
            sim = cosine_similarity(norm_emb, l2_normalize(other))
            if sim >= DUPLICATE_FACE_THRESHOLD:
                raise HTTPException(
                    status_code=409,
                    detail=f"Face already registered to another account ({u.email})."
                )

        face_str = json.dumps(norm_emb)
    
    # Create user
    user = User(
        name=user_data.name,
        email=user_data.email,
        hashed_password=hashed,
        role=user_data.role,
        student_id=student_id,
        student_id_edited=False,
        face_embedding=face_str
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # Create token
    token = create_access_token({"user_id": user.id, "role": user.role, "student_id": student_id})
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user.role,
        "name": user.name,
        "student_id": student_id
    }
@app.post("/auth/login", response_model=Token)
def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """Login user"""
    user = db.query(User).filter(User.email == credentials.email).first()
    if not user or not pwd_context.verify(credentials.password[:72], user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid email or password")
    
    token = create_access_token({"user_id": user.id, "role": user.role, "student_id": user.student_id})
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user.role,
        "name": user.name,
        "student_id": user.student_id
    }

@app.get("/user/info")
def get_user_info(token_data: dict = Depends(verify_token), db: Session = Depends(get_db)):
    """Get current user information including student ID and edit status"""
    user = db.query(User).filter(User.id == token_data["user_id"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    response = {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": user.role,
        "student_id": user.student_id
    }
    
    # Add edit status for students
    if user.role == "student":
        response["student_id_edited"] = user.student_id_edited
        response["can_edit_student_id"] = not user.student_id_edited
    
    return response
@app.put("/user/student-id")
def update_student_id(
    update_data: StudentIdUpdate,
    token_data: dict = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Update student ID (can only be done once)"""
    user = db.query(User).filter(User.id == token_data["user_id"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Only students can update their student ID
    if user.role != "student":
        raise HTTPException(status_code=403, detail="Only students have student IDs")
    
    # Check if already edited
    if user.student_id_edited:
        raise HTTPException(
            status_code=403,
            detail="Student ID has already been edited once. No further changes allowed."
        )
    
    # Validate new student ID
    new_id = update_data.new_student_id.strip()
    if not new_id.isdigit() or len(new_id) != 8:
        raise HTTPException(status_code=400, detail="Student ID must be exactly 8 digits")
    
    # Check if new ID is already taken by another user
    existing = db.query(User).filter(
        User.student_id == new_id,
        User.id != user.id
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="This Student ID is already registered")
    
    # Update student ID and mark as edited
    user.student_id = new_id
    user.student_id_edited = True
    db.commit()
    
    return {
        "message": "Student ID updated successfully",
        "student_id": new_id,
        "can_edit_again": False
    }
@app.post("/courses/create")
def create_course(course_data: CourseCreate, token_data: dict = Depends(verify_token), db: Session = Depends(get_db)):
    """Create a new course (professors only)"""
    if token_data["role"] != "professor":
        raise HTTPException(status_code=403, detail="Only professors can create courses")
    
    join_code = generate_join_code()
    
    # Ensure join code is unique
    while db.query(Course).filter(Course.code == join_code).first():
        join_code = generate_join_code()
    
    course = Course(
        name=course_data.name,
        code=join_code,
        professor_id=token_data["user_id"]
    )
    db.add(course)
    db.commit()
    db.refresh(course)
    
    return {"course_id": course.id, "name": course.name, "join_code": join_code}

@app.get("/courses/my-taught")
def get_my_taught_courses(token_data: dict = Depends(verify_token), db: Session = Depends(get_db)):
    """Get courses taught by the current professor"""
    if token_data["role"] != "professor":
        raise HTTPException(status_code=403, detail="Only professors can access this endpoint")
    
    courses = db.query(Course).filter(Course.professor_id == token_data["user_id"]).all()
    
    return [
        {
            "id": c.id,
            "name": c.name,
            "code": c.code,
            "student_count": len(c.students)
        }
        for c in courses
    ]

@app.post("/courses/enroll")
def enroll_in_course(enroll_data: CourseEnroll, token_data: dict = Depends(verify_token), db: Session = Depends(get_db)):
    """Enroll in a course using join code"""
    if token_data["role"] != "student":
        raise HTTPException(status_code=403, detail="Only students can enroll in courses")
    
    course = db.query(Course).filter(Course.code == enroll_data.join_code.upper()).first()
    if not course:
        raise HTTPException(status_code=404, detail="Invalid join code")
    
    student = db.query(User).filter(User.id == token_data["user_id"]).first()
    
    # Check if already enrolled
    if student in course.students:
        raise HTTPException(status_code=400, detail="Already enrolled in this course")
    
    course.students.append(student)
    db.commit()
    
    return {"course_id": course.id, "course_name": course.name}

@app.get("/courses/enrolled")
def get_enrolled_courses(token_data: dict = Depends(verify_token), db: Session = Depends(get_db)):
    """Get courses the current student is enrolled in"""
    if token_data["role"] != "student":
        raise HTTPException(status_code=403, detail="Only students can access this endpoint")
    
    student = db.query(User).filter(User.id == token_data["user_id"]).first()
    
    return [
        {
            "id": c.id,
            "name": c.name
        }
        for c in student.enrolled_courses
    ]

@app.get("/courses/{course_id}/roster")
def get_course_roster(course_id: int, token_data: dict = Depends(verify_token), db: Session = Depends(get_db)):
    """Get list of enrolled students with their IDs for a course (professors only)"""
    if token_data["role"] != "professor":
        raise HTTPException(status_code=403, detail="Only professors can access rosters")
    
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    if course.professor_id != token_data["user_id"]:
        raise HTTPException(status_code=403, detail="Not authorized for this course")
    
    return [
        {
            "id": s.id,
            "name": s.name,
            "student_id": s.student_id,
            "email": s.email
        }
        for s in course.students
    ]

# ... existing code ...

@app.get("/courses/{course_id}/roster")
def get_course_roster(course_id: int, token_data: dict = Depends(verify_token), db: Session = Depends(get_db)):
    """Get list of enrolled students with their IDs for a course (professors only)"""
    # ... existing code ...

# ADD THE NEW DELETE ENDPOINT HERE:
@app.delete("/courses/{course_id}")
def delete_course(course_id: int, token_data: dict = Depends(verify_token), db: Session = Depends(get_db)):
    """Delete a course and all related data (professors only)"""
    if token_data["role"] != "professor":
        raise HTTPException(status_code=403, detail="Only professors can delete courses")
    
    # Find the course
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    # Verify ownership
    if course.professor_id != token_data["user_id"]:
        raise HTTPException(status_code=403, detail="Not authorized to delete this course")
    
    # Get all sessions for this course
    sessions = db.query(Session).filter(Session.course_id == course_id).all()
    session_ids = [s.id for s in sessions]
    
    # Delete all attendance records for these sessions
    if session_ids:
        db.query(Attendance).filter(Attendance.session_id.in_(session_ids)).delete(synchronize_session=False)
    
    # Delete all sessions
    db.query(Session).filter(Session.course_id == course_id).delete(synchronize_session=False)
    
    # Remove all student enrollments (many-to-many relationship)
    course.students.clear()
    
    # Delete the course itself
    db.delete(course)
    db.commit()
    
    return {"message": f"Course '{course.name}' and all related data deleted successfully"}

    
@app.post("/sessions/start")
def start_session(session_data: SessionStart, token_data: dict = Depends(verify_token), db: Session = Depends(get_db)):
    """Start an attendance session (professors only)"""
    if token_data["role"] != "professor":
        raise HTTPException(status_code=403, detail="Only professors can start sessions")
    
    # Verify course ownership
    course = db.query(Course).filter(Course.id == session_data.course_id).first()
    if not course or course.professor_id != token_data["user_id"]:
        raise HTTPException(status_code=403, detail="Not authorized for this course")
    
    # End any active sessions for this course
    db.query(Session).filter(Session.course_id == session_data.course_id, Session.is_active == True).update(
        {"is_active": False, "end_time": datetime.now(ET)}
    )
    
    # Create new session
    session = Session(
        course_id=session_data.course_id,
        late_after_minutes=session_data.late_after_minutes,
        absent_after_minutes=session_data.absent_after_minutes
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    
    return {"session_id": session.id, "course_name": course.name}

@app.post("/sessions/stop/{session_id}")
def stop_session(session_id: int, token_data: dict = Depends(verify_token), db: Session = Depends(get_db)):
    """Stop an attendance session (professors only)"""
    if token_data["role"] != "professor":
        raise HTTPException(status_code=403, detail="Only professors can stop sessions")
    
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Verify course ownership
    course = db.query(Course).filter(Course.id == session.course_id).first()
    if course.professor_id != token_data["user_id"]:
        raise HTTPException(status_code=403, detail="Not authorized for this session")
    
    session.is_active = False
    session.end_time = datetime.now(ET)
    
    # Mark absent students
    enrolled_students = course.students
    checked_in_students = db.query(Attendance.student_id).filter(Attendance.session_id == session_id).all()
    checked_in_ids = [s[0] for s in checked_in_students]
    
    for student in enrolled_students:
        if student.id not in checked_in_ids:
            attendance = Attendance(
                session_id=session_id,
                student_id=student.id,
                status="absent",
                timestamp=datetime.now(ET)  # Add explicit timestamp
            )
            db.add(attendance)
    
    db.commit()
    
    return {"message": "Session stopped", "absent_marked": len(enrolled_students) - len(checked_in_ids)}

@app.get("/sessions/active")
def get_active_session(course_id: int, db: Session = Depends(get_db)):
    """Check if there's an active session for a course"""
    session = db.query(Session).filter(
        Session.course_id == course_id, 
        Session.is_active == True
    ).first()
    
    if session:
        return {"active": True, "session_id": session.id}
    return {"active": False}

@app.post("/checkin/with-id")
def checkin_with_student_id(checkin_data: CheckInWithStudentId, db: Session = Depends(get_db)):
    """Check in using student ID and face verification (used in professor's console)"""
    # Get the active session for the course
    session = db.query(Session).filter(
        Session.course_id == checkin_data.course_id,
        Session.is_active == True
    ).first()
    
    if not session:
        raise HTTPException(status_code=400, detail="No active session for this course")
    
    # Find student by student ID
    student = db.query(User).filter(User.student_id == checkin_data.student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Invalid student ID")
    
    # Check if student is enrolled in the course
    course = db.query(Course).filter(Course.id == checkin_data.course_id).first()
    if student not in course.students:
        raise HTTPException(status_code=403, detail="Student not enrolled in this course")
    
    # Check if already checked in
    existing = db.query(Attendance).filter(
        Attendance.session_id == session.id,
        Attendance.student_id == student.id
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Already checked in")
    
    # Verify face
    if not student.face_embedding:
        raise HTTPException(status_code=400, detail="No face data on file for this student")
    
    stored_embedding = json.loads(student.face_embedding)
    normalized_input = l2_normalize(checkin_data.face_embedding)
    similarity = cosine_similarity(normalized_input, stored_embedding)
    
    if similarity < FACE_MATCH_THRESHOLD:
        raise HTTPException(status_code=403, detail=f"Face verification failed!")
    
    # Determine status based on time
    # Determine status based on time
    start_time = session.start_time if session.start_time.tzinfo else session.start_time.replace(tzinfo=ET)
    elapsed_minutes = (datetime.now(ET) - start_time).total_seconds() / 60
    status = "present"
    if elapsed_minutes > session.late_after_minutes:
        status = "late"
    if elapsed_minutes > session.absent_after_minutes:
        status = "absent"
    
    # Record attendance
    attendance = Attendance(
        session_id=session.id,
        student_id=student.id,
        status=status,
        confidence=similarity
    )
    db.add(attendance)
    db.commit()
    
    return {
        "student_name": student.name,
        "status": status,
        "confidence": f"{similarity:.2%}"
    }

@app.post("/checkin")
def checkin(checkin_data: CheckIn, token_data: dict = Depends(verify_token), db: Session = Depends(get_db)):
    """Original check-in endpoint (kept for backward compatibility)"""
    if token_data["role"] != "student":
        raise HTTPException(status_code=403, detail="Only students can check in")
    
    session = db.query(Session).filter(
        Session.course_id == checkin_data.course_id,
        Session.is_active == True
    ).first()
    
    if not session:
        raise HTTPException(status_code=400, detail="No active session for this course")
    
    student = db.query(User).filter(User.id == token_data["user_id"]).first()
    
    # Check enrollment
    course = db.query(Course).filter(Course.id == checkin_data.course_id).first()
    if student not in course.students:
        raise HTTPException(status_code=403, detail="Not enrolled in this course")
    
    # Check if already checked in
    existing = db.query(Attendance).filter(
        Attendance.session_id == session.id,
        Attendance.student_id == student.id
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Already checked in")
    
    # Verify face
    stored_embedding = json.loads(student.face_embedding)
    normalized_input = l2_normalize(checkin_data.face_embedding)
    similarity = cosine_similarity(normalized_input, stored_embedding)
    
    if similarity < FACE_MATCH_THRESHOLD:
        raise HTTPException(status_code=403, detail=f"Face verification failed")
    
    # Determine status
    start_time = session.start_time if session.start_time.tzinfo else session.start_time.replace(tzinfo=ET)
    elapsed_minutes = (datetime.now(ET) - start_time).total_seconds() / 60
    status = "present"
    if elapsed_minutes > session.late_after_minutes:
        status = "late"
    if elapsed_minutes > session.absent_after_minutes:
        status = "absent"
    
    attendance = Attendance(
        session_id=session.id,
        student_id=student.id,
        status=status,
        confidence=similarity
    )
    db.add(attendance)
    db.commit()
    
    return {"status": status}

@app.get("/attendance/live/{course_id}")
def get_live_attendance(course_id: int, token_data: dict = Depends(verify_token), db: Session = Depends(get_db)):
    """Get live attendance for active session"""
    if token_data["role"] != "professor":
        raise HTTPException(status_code=403, detail="Only professors can view live attendance")
    
    # Get active session
    session = db.query(Session).filter(
        Session.course_id == course_id,
        Session.is_active == True
    ).first()
    
    if not session:
        return []
    
    # Get attendance records
    records = db.query(Attendance).filter(Attendance.session_id == session.id).all()
    
    return [
        {
            "student_name": r.student.name,
            "student_id": r.student.student_id,
            "timestamp": r.timestamp.isoformat(),
            "status": r.status
        }
        for r in records
    ]

@app.get("/attendance/history")
def get_attendance_history(token_data: dict = Depends(verify_token), db: Session = Depends(get_db)):
    """Get attendance history for professor's courses"""
    if token_data["role"] != "professor":
        raise HTTPException(status_code=403, detail="Only professors can view attendance history")
    
    # Get all courses taught by this professor
    courses = db.query(Course).filter(Course.professor_id == token_data["user_id"]).all()
    course_ids = [c.id for c in courses]
    
    # Get all sessions for these courses
    sessions = db.query(Session).filter(Session.course_id.in_(course_ids)).all()
    session_ids = [s.id for s in sessions]
    
    # Get all attendance records
    records = db.query(Attendance).filter(Attendance.session_id.in_(session_ids)).order_by(Attendance.timestamp.desc()).all()
    
    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat(),
            "student_name": r.student.name,
            "student_id": r.student.student_id,
            "course_name": r.session.course.name,
            "course_id": r.session.course.id,
            "status": r.status
        }
        for r in records
    ]

@app.get("/attendance/my-history")
def get_my_attendance_history(token_data: dict = Depends(verify_token), db: Session = Depends(get_db)):
    """Get attendance history for a student"""
    if token_data["role"] != "student":
        raise HTTPException(status_code=403, detail="Only students can view their attendance")
    
    records = db.query(Attendance).filter(Attendance.student_id == token_data["user_id"]).order_by(Attendance.timestamp.desc()).all()
    
    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat(),
            "course_name": r.session.course.name,
            "course_id": r.session.course.id,
            "status": r.status
        }
        for r in records
    ]

@app.delete("/students/{student_id}/data")
def delete_student_data(student_id: str, token_data: dict = Depends(verify_token), db: Session = Depends(get_db)):
    """Delete student enrollments and face data (professors only)"""
    if token_data["role"] != "professor":
        raise HTTPException(status_code=403, detail="Only professors can delete student data")
    
    # Find student by student ID
    student = db.query(User).filter(User.student_id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    # Get professor's courses
    professor_courses = db.query(Course).filter(Course.professor_id == token_data["user_id"]).all()
    professor_course_ids = [c.id for c in professor_courses]
    
    # Remove student from professor's courses only
    for course in professor_courses:
        if student in course.students:
            course.students.remove(student)
    
    # Delete attendance records for professor's courses
    sessions_to_check = db.query(Session).filter(Session.course_id.in_(professor_course_ids)).all()
    session_ids = [s.id for s in sessions_to_check]
    db.query(Attendance).filter(
        Attendance.student_id == student.id,
        Attendance.session_id.in_(session_ids)
    ).delete()
    
    # Clear face embedding (optional - only if professor owns ALL courses the student is in)
    # For safety, we'll only clear if the student is not enrolled in any other courses
    if not student.enrolled_courses:
        student.face_embedding = None
    
    db.commit()
    
    return {"message": f"Student {student.name} removed from your courses"}

@app.delete("/students/all/data")
def delete_all_students_data(token_data: dict = Depends(verify_token), db: Session = Depends(get_db)):
    """Delete ALL students' face embeddings and course enrollments (professors only)"""
    if token_data["role"] != "professor":
        raise HTTPException(status_code=403, detail="Only professors can delete student data")
    
    # Get all students (users with role='student')
    all_students = db.query(User).filter(User.role == 'student').all()
    
    if not all_students:
        return {"message": "No students found in the system", "deleted_count": 0}
    
    deleted_count = 0
    
    for student in all_students:
        # Clear face embedding
        if student.face_embedding:
            student.face_embedding = None
            deleted_count += 1
        
        # Remove from all courses
        student.enrolled_courses.clear()
        
        # Delete all attendance records for this student
        db.query(Attendance).filter(Attendance.student_id == student.id).delete()
    
    db.commit()
    
    return {
        "message": f"Successfully deleted face embeddings and enrollments for all students",
        "deleted_count": deleted_count
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
