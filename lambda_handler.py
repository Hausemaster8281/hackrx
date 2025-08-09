from mangum import Mangum
from app.main import app

# Lambda handler
handler = Mangum(app)
